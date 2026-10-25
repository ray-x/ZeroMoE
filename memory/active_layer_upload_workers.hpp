// active_layer_upload_workers.hpp
//
// 2026-08-28: background worker-thread pool that takes an active-layer H2D
// upload's cudaMemcpyAsync OFF the decode thread entirely, for exactly the
// population that pays for it: sources whose RAM-tier slab is NOT already
// covered by ram_pinned_budget_gb's persistent pin (MoEExpertBlock::
// slab_pinned == false). A pinned source's inline upload_expert_active_async
// call is already near-free (~0.005ms/call, measured) -- routing THOSE
// through a worker would only add cross-thread handoff overhead for a call
// that was never the problem. An unpinned source's inline call, by contrast,
// secretly blocks the calling thread for ~1.7ms (a CUDA-driver-internal
// staged copy, confirmed via real nvidia-smi PCIe telemetry to NOT be a
// bandwidth cost -- see t_active_layer_upload_async_unpinned's own comment,
// perf_timer.hpp) -- multiplied across thousands of decode-time calls/
// request, this IS active_layer_prewarm_issue's real, CPU-bound cost.
//
// Two-phase split across threads, mirroring GpuTopology::claim_active_layer_
// slot_for_worker's own declaration comment:
//   1. DECODE THREAD (fast, synchronous, ~microseconds): ram_pin the RAM
//      source + fresh identity re-check (existing TOCTOU-safe pattern, see
//      execution_layer.cpp's decode_async_prewarm loop), then claim_active_
//      layer_slot_for_worker (victim selection only -- a short cache_mutex
//      critical section, no copy/anything expensive). On success, hands the
//      claimed slot + already-pinned RAM source off to this pool via
//      try_submit and returns immediately -- no waiting.
//   2. WORKER THREAD (slow, off the decode thread): pops the item, does the
//      plain unpinned cudaMemcpyAsync + cudaEventRecord (see process()'s own
//      comment for why this is deliberately NOT a cudaHostRegister'd
//      transfer -- a real, hardware-confirmed crash was traced to exactly
//      that), then two brief, NEVER-NESTED locked hand-offs: GpuTopology::
//      finalize_worker_dispatch (flips copy_dispatched=true) and
//      RAMSafetyManager::ram_mark_pending_copy_and_unpin (transfers the RAM
//      source's protection from ram_temp_pin_count to pending_copy_event,
//      self-healing/polled, and unpins).
//
// Off by default (EngineConfig::active_layer_upload_worker_threads = 0) --
// zero risk when off: no thread pool is constructed, every call site keeps
// using its existing inline path unconditionally, matching this project's
// established "opt-in, zero cost when off" convention for every prior
// active-layer feature (decode_async_prewarm, speculative_promote_to_
// active_layer, etc.).
//
// REAL HARDWARE HISTORY (2026-08-28) -- THREE real, independent
// correctness bugs found and fixed before this pool was trustworthy, all
// via real hardware, not guessed. This took several rounds; each earlier
// round's fix is kept below as the historical record of how the
// investigation actually progressed, since two of them looked complete at
// the time and were not.
//
// Bug 1 (pending_host_unregisters starvation): the FIRST version of this
// file deferred cudaHostUnregister via GpuTopology::pending_host_
// unregisters (the same queue upload_expert_async's own Alt-A conversion
// uses). That queue is only reaped opportunistically from upload_expert_
// async's own call site (plus the destructor) -- in a session dominated by
// this pool's two callers, upload_expert_async can fire rarely or never,
// so entries pushed from here could sit unreaped indefinitely, letting a
// RAM-tier slab get freed/realloc'd (engine.hpp's ensure_slab_capacity,
// which only guards its own persistent slab_pinned flag) while still
// page-locked under a registration nobody had unregistered yet. Fixed
// (at the time) by making register/unregister self-contained per worker
// thread. Real NaN-cascade corruption on real hardware; fix confirmed the
// specific symptom gone, but a second, different bug (below) was still
// live.
//
// Bug 2: GpuTopology::upload_expert_active's own real-hardware duplicate-
// transfer pre-check (added earlier the same session, before claim_active_
// layer_slot_for_worker/copy_dispatched existed) scanned active_layer_
// slots for a matching identity and, if found with upload_ready==false,
// treated slot.upload_event as a genuinely-pending recording and called
// cudaEventSynchronize on it -- but for a slot claimed by this pool and
// not yet dispatched, that event object exists (pre-allocated at pool
// creation) but was never re-recorded for THIS claim; cudaEventSynchronize
// on a stale/unrecorded event returns immediately, spuriously marking a
// still-stale slot "ready". Fixed by adding the same copy_dispatched guard
// this pre-check's two siblings (find_active_expert, cleanup_active_layer_
// area) already had. Real, correct fix -- but a real, controlled two-
// prompts-in-one-process test (see Bug 3) still crashed after this landed,
// meaning Bugs 1 and 2 were both real but neither was the crash's actual
// cause.
//
// Bug 3 (the ACTUAL root cause): every real-hardware crash was a genuine
// GPU-level fault (NVRM Xid 31, "MMU Fault: ENGINE CE2 ... faulted @
// 0x0_00000000", confirmed via `journalctl -k`, not just an application-
// level symptom), reliably reproduced by running TWO prompts in one
// process (normal REPL usage -- Bugs 1/2's own "verified" single-prompt-
// per-process testing never exercised this). A methodical elimination
// (each step confirmed on real hardware): (a) held cache_mutex around the
// register+cudaMemcpyAsync+cudaEventRecord sequence, matching upload_
// expert_async's own only proven-safe background-thread-CUDA pattern --
// still crashed. (b) active_layer_upload_worker_threads=1 (zero worker-
// vs-worker concurrency possible) -- still crashed, ruling out a race
// between worker threads specifically. (c) A controlled experiment
// skipping cudaHostRegister entirely (falling straight through to a plain
// unpinned cudaMemcpyAsync, matching what upload_expert_active_async
// already did for an unpinned source before this pool existed) ran the
// exact same crashing scenario clean, twice in a row -- decisive. Root
// cause, as far as it can be pinned down without NVIDIA's own driver
// source: this pool's cudaHostRegister call was the ONLY thing that ever
// introduced per-call host-memory (un)registration for active-layer
// uploads at all (the original upload_expert_active_async never
// registers); ensure_slab_capacity's OWN persistent-pin cudaHostRegister
// (EngineConfig::ram_pinned_budget_gb) runs under a different lock
// (ram_mutex, not cache_mutex) and so can always run genuinely
// concurrently with this pool's own register call regardless of (a) --
// doing so at real volume (tens of thousands of calls/session) under this
// box's own independently-observed tight host-memory budget (free RAM
// falling to 3-5GB) is the most likely trigger, though the exact NVIDIA-
// driver-internal mechanism is not confirmed. FIX: this pool no longer
// calls cudaHostRegister/cudaHostUnregister AT ALL -- every transfer is a
// plain unpinned cudaMemcpyAsync, exactly matching the original inline
// path's own fallback behavior for an unpinned source, just issued from a
// worker thread instead of the decode thread.
//
// VERIFIED ON REAL HARDWARE, properly this time: the exact scenario that
// crashed 4 times in a row with cudaHostRegister enabled (2 workers, both
// callers, two prompts in one process) ran clean, with zero [ERROR] lines
// and zero Xid faults, across 4 separate runs after the fix -- three
// two-prompt sessions and one three-prompt session, each confirmed via
// `journalctl -k` showing nothing new. Settled default: active_layer_
// upload_worker_threads = 2 (see configs/obsolete/glm52_3090_5060_sdd2.toml).
//
// Throughput: decode-only tok/s (40 tokens / (request_wall - prefill_wall)
// -- NOT the raw [REQUEST] line's own combined prefill+decode rate, which
// is a different, incomparable number against this codebase's historical
// decode-only baselines, a mistake made and caught mid-session) was flat
// vs. the pool-disabled baseline in the original (pre-fix, register-
// enabled) testing; not yet re-measured against the register-removed
// version specifically. Correctness, not speed, is this entry's claim.

#pragma once

#include "ram_manager.hpp"
#include "perf_timer.hpp"
#include <cuda_runtime.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <atomic>

struct ActiveLayerUploadWorkItem {
    ExpertLocation* slot = nullptr;   // already claimed via claim_active_layer_slot_for_worker
    MoEExpertBlock* blk = nullptr;    // RAM-tier source block, already ram_pin()'d by the submitter
    const uint8_t* host_data = nullptr;  // = blk->slab_buffer, captured at submit time (stable:
                                          // ram_pin excludes this slot from every victim-selection
                                          // scan until ram_mark_pending_copy_and_unpin unpins it)
    size_t host_bytes = 0;                // = blk->slab_capacity, captured at submit time
};

class ActiveLayerUploadWorkers {
public:
    // queue_capacity: generous, bounded -- overflow (try_submit returning
    // false) is handled by the caller falling back cleanly (abandon the
    // claim, ram_unpin the source, let the existing synchronous path or a
    // later decode-time miss pick this expert up instead), never by
    // blocking the decode thread waiting for room.
    ActiveLayerUploadWorkers(GpuTopology& topo, RAMSafetyManager& ram_mgr,
                              int n_workers, size_t queue_capacity = 64)
        : topo_(topo), ram_mgr_(ram_mgr), queue_capacity_(queue_capacity) {
        workers_.reserve((size_t)std::max(0, n_workers));
        for (int i = 0; i < n_workers; ++i) {
            workers_.emplace_back([this]() { this->worker_loop(); });
        }
    }

    ~ActiveLayerUploadWorkers() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            shutting_down_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    ActiveLayerUploadWorkers(const ActiveLayerUploadWorkers&) = delete;
    ActiveLayerUploadWorkers& operator=(const ActiveLayerUploadWorkers&) = delete;

    bool enabled() const { return !workers_.empty(); }

    // Non-blocking. Returns false if the queue is at capacity -- caller MUST
    // abandon the already-claimed slot (GpuTopology::abandon_worker_claim)
    // and ram_unpin the source block itself in that case; this function
    // never does either on behalf of a rejected item, since the caller is
    // the one holding the fresh identity-checked references.
    bool try_submit(const ActiveLayerUploadWorkItem& item) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (queue_.size() >= queue_capacity_) {
                perf().n_active_layer_worker_queue_full.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            queue_.push_back(item);
        }
        cv_.notify_one();
        perf().n_active_layer_worker_submitted.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

private:
    // 2026-08-29: drains a small NON-BLOCKING batch of already-queued items
    // (up to kFinalizeBatchCap) after the initial blocking wait, instead of
    // processing strictly one item per wake-up -- purely to reduce how
    // many separate finalize_worker_dispatch_batch calls (and so cache_
    // mutex acquisitions) this thread makes, not to add latency: if only
    // one item is queued, this behaves identically to before (process it,
    // finalize it, loop). Never waits for MORE items to arrive beyond
    // what's already sitting in the queue at the moment of the check.
    static constexpr size_t kFinalizeBatchCap = 8;

    void worker_loop() {
        std::vector<ExpertLocation*> finalize_batch;
        finalize_batch.reserve(kFinalizeBatchCap);
        for (;;) {
            ActiveLayerUploadWorkItem item;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this]() { return shutting_down_ || !queue_.empty(); });
                if (shutting_down_ && queue_.empty()) return;
                if (queue_.empty()) continue;  // spurious wake during shutdown race
                item = queue_.front();
                queue_.pop_front();
            }
            finalize_batch.clear();
            if (ExpertLocation* slot = process_item(item)) finalize_batch.push_back(slot);
            // Drain whatever else is ALREADY queued right now (no extra
            // waiting) -- each item still gets its own full process_item
            // (real per-item cudaMemcpyAsync + cudaStreamSynchronize +
            // status check, unchanged), only the final bookkeeping is
            // batched.
            while (finalize_batch.size() < kFinalizeBatchCap) {
                ActiveLayerUploadWorkItem next;
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    if (queue_.empty()) break;
                    next = queue_.front();
                    queue_.pop_front();
                }
                if (ExpertLocation* slot = process_item(next)) finalize_batch.push_back(slot);
            }
            topo_.finalize_worker_dispatch_batch(finalize_batch);
        }
    }

    // 2026-08-29: returns the slot on a normal completion (copy issued,
    // waited-on, status-checked -- but NOT yet flipped copy_dispatched=
    // true) so worker_loop can batch several already-completed items'
    // finalize_worker_dispatch calls into one cache_mutex acquisition
    // (see GpuTopology::finalize_worker_dispatch_batch's own comment for
    // why). Returns nullptr for the invalid-item early-return path, which
    // still finalizes itself immediately (rare error path, not worth
    // batching, and item.blk's ram_unpin needs to happen right away
    // regardless). This does NOT change when a copy is considered done --
    // cudaStreamSynchronize below still runs per-item, synchronously, on
    // this same worker thread, exactly as before; only the "tell other
    // readers this slot is ready" bookkeeping is deferred to the caller.
    ExpertLocation* process_item(const ActiveLayerUploadWorkItem& item) {
        ExpertLocation* slot = item.slot;
        if (!slot) return nullptr;
        // 2026-08-28: defensive hardening added while chasing a real,
        // reproduced-twice hardware crash (NVRM Xid 31 MMU fault, a copy
        // engine reading from address 0x0 -- see the wait/unregister
        // reordering fix below, and its own comment, for the race that was
        // found and fixed alongside this). Neither of these two checks was
        // present before; if either ever fires, it means something UPSTREAM
        // of this pool (a stale claim, a freed RAM slab) is already wrong --
        // failing loudly and safely here, instead of feeding a null/invalid
        // pointer into cudaMemcpyAsync and silently corrupting the CUDA
        // context (which then surfaces as a confusing, unrelated failure in
        // some completely different call site much later), is strictly
        // better even if this specific check never ends up being the one
        // that fires in practice.
        if (!item.host_data || item.host_bytes == 0 || !slot->device_ptr) {
            LOG_STREAM_ERROR() << "[GPU] ActiveLayerUploadWorkers: invalid work item (host_data="
                               << (const void*)item.host_data << " host_bytes=" << item.host_bytes
                               << " device_ptr=" << (const void*)slot->device_ptr
                               << ") -- refusing to copy, releasing claim\n";
            topo_.finalize_worker_dispatch(slot, nullptr, /*register_succeeded=*/false);
            if (item.blk) ram_mgr_.ram_mark_pending_copy_and_unpin(*item.blk, nullptr);
            perf().n_active_layer_worker_invalid_item.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        cudaSetDevice(slot->gpu_id);

        // 2026-08-28 REAL ROOT CAUSE FOUND ON REAL HARDWARE (after two
        // other real fixes -- a pending_host_unregisters starvation
        // hazard, and a TOCTOU reorder -- and a lock-scoping change, none
        // of which stopped the crash): this pool used to call
        // cudaHostRegister/cudaHostUnregister per item, exactly like
        // upload_expert_async's own Alt-A conversion. That per-call
        // register/unregister CYCLE ITSELF is the trigger, not locking or
        // thread count. Proof, in order: (1) locking cudaHostRegister/
        // cudaMemcpyAsync/cudaEventRecord under cache_mutex (matching
        // upload_expert_async's own only proven-safe background-thread-
        // CUDA pattern) did NOT stop the crash. (2) active_layer_upload_
        // worker_threads=1 (zero worker-vs-worker concurrency possible)
        // STILL crashed -- ruling out a race between worker threads
        // specifically. (3) A controlled experiment skipping cudaHost
        // Register entirely (falling straight through to the plain
        // unpinned cudaMemcpyAsync below, every time) ran the EXACT same
        // crashing scenario (2 workers, both callers, two prompts in one
        // process) clean, twice in a row -- decisive.
        //
        // Root cause, as far as it can be pinned down without NVIDIA's own
        // driver source: this pool's cudaHostRegister call is protected by
        // cache_mutex, but ensure_slab_capacity's OWN persistent-pin
        // cudaHostRegister call (EngineConfig::ram_pinned_budget_gb,
        // engine.hpp) runs under a COMPLETELY DIFFERENT lock (ram_mutex,
        // via whichever RAM-tier caller triggers a slab load) -- so even
        // with this pool's own call properly serialized against ITSELF,
        // it can still run genuinely concurrently with that unrelated
        // registration. Active-layer uploads never called cudaHostRegister
        // AT ALL before this pool existed (the original upload_expert_
        // active_async just does a plain, possibly-slow-but-safe
        // cudaMemcpyAsync on unpinned memory) -- this pool was the ONLY
        // thing that ever introduced concurrent host-memory (un)registration
        // against this specific combination of GPU/driver state, and doing
        // so at real volume (tens of thousands of calls/session) under
        // this box's own independently-observed tight host-memory budget
        // (free RAM falling to 3-5GB) is judged the most likely trigger,
        // though the exact NVIDIA-driver-internal mechanism is not
        // confirmed. FIX: never call cudaHostRegister/cudaHostUnregister
        // from this pool at all -- always issue the plain unpinned
        // cudaMemcpyAsync below (same driver-internal staged-copy fallback
        // the original upload_expert_active_async already relied on for
        // an unpinned source; slower per-call than a genuine pinned DMA,
        // but off the decode thread either way, and PROVEN safe where the
        // register/unregister cycle was not). VERIFIED ON REAL HARDWARE:
        // the exact scenario that crashed 4 times in a row with register
        // enabled ran clean 2 times in a row with it removed.
        size_t copy_bytes = std::min(item.host_bytes, slot->size);
        auto& dev = topo_.devices[slot->gpu_id];
        // Same stream-selection rule as upload_expert_active_async's own --
        // see GpuDevice::speculative_prefetch_stream's comment for why a
        // speculative fill gets its own stream, separate from active_layer_
        // stream's more time-critical (decode_async_prewarm, reactive)
        // traffic.
        cudaStream_t issue_stream = slot->pilot_speculative_fill
            ? dev.speculative_prefetch_stream : dev.active_layer_stream;
        cudaError_t copy_err = cudaMemcpyAsync(slot->device_ptr, item.host_data, copy_bytes,
                                                cudaMemcpyHostToDevice, issue_stream);
        if (copy_err != cudaSuccess) {
            // cudaMemcpyAsync's own immediate return only ever reports a
            // LAUNCH-time error (bad stream/direction) -- a bad POINTER
            // surfaces later, asynchronously, once the copy engine actually
            // executes it. Logged here for the launch-time case; the
            // execution-time case is caught below by the unconditional
            // synchronize+status check.
            LOG_STREAM_ERROR() << "[GPU] ActiveLayerUploadWorkers: cudaMemcpyAsync launch failed for layer="
                               << slot->layer_idx << " expert=" << slot->expert_id << ": "
                               << cudaGetErrorString(copy_err) << "\n";
        }
        if (slot->upload_event) cudaEventRecord(slot->upload_event, issue_stream);
        // 2026-08-28: unconditional synchronous status check, added while
        // chasing the crash above -- makes any future execution-time fault
        // on this call surface HERE, immediately, with full context
        // (layer/expert/bytes/pointers), instead of silently corrupting
        // the CUDA context and only showing up as a confusing, unrelated
        // "unspecified launch failure" somewhere else much later. Cheap:
        // this worker thread was going to do useful work either way, and
        // this costs nothing on the decode thread.
        cudaError_t sync_err = cudaStreamSynchronize(issue_stream);
        if (sync_err != cudaSuccess) {
            LOG_STREAM_ERROR() << "[GPU] ActiveLayerUploadWorkers: async copy FAILED for layer="
                               << slot->layer_idx << " expert=" << slot->expert_id << " bytes="
                               << copy_bytes << " host_data=" << (const void*)item.host_data
                               << " device_ptr=" << (const void*)slot->device_ptr << ": "
                               << cudaGetErrorString(sync_err) << "\n";
            perf().n_active_layer_worker_copy_fail.fetch_add(1, std::memory_order_relaxed);
        }

        // 2026-08-31 CORRECTNESS FIX. Previously BOTH failure branches above
        // only LOGGED, then fell through and published this slot as if the
        // copy had succeeded -- so a slot still holding uninitialized VRAM
        // entered the active-layer pool indistinguishable from a good one,
        // and the very next dispatch ran a real matmul against it.
        //
        // That is how one failed copy produced 939 errors and garbled output
        // on a real run: the weights are QUANTIZED, so uninitialized bytes
        // are read as fp16 superblock scales, and ~3.1% of random 16-bit
        // patterns decode to Inf/NaN (exponent all-1s). A single NaN weight
        // poisons its whole dot-product output; RMSNorm then divides by a
        // NaN-poisoned row norm, turning partial corruption (965/6144) into
        // total (6144/6144) and propagating it through every later layer and
        // token. Measured NaN rate matched that prediction closely.
        //
        // So: on ANY copy failure, INVALIDATE the slot instead of publishing
        // it. expert_id=-1 makes find_active_expert miss it, and a miss is
        // an already-proven-safe path -- dispatch falls back to a fresh
        // fetch (or CPU), which is slower but CORRECT. Returning nullptr
        // also keeps it out of the finalize batch, so copy_dispatched stays
        // false and no other thread can observe it as ready.
        //
        // This is deliberately independent of WHY the copy failed: whatever
        // the root cause, computing on a slot whose upload failed can never
        // be right, so this closes the blast radius rather than any one bug.
        if (copy_err != cudaSuccess || sync_err != cudaSuccess) {
            slot->expert_id = -1;
            slot->upload_ready = false;
            if (slot->temp_pin_count > 0) --slot->temp_pin_count;
            // Release the RAM-tier source hold on the same path a successful
            // copy uses -- nullptr event because there is no in-flight copy
            // left to wait on (both failure modes are already synchronized:
            // a launch failure never started one, and sync_err means the
            // stream has already drained).
            ram_mgr_.ram_mark_pending_copy_and_unpin(*item.blk, nullptr);
            perf().n_active_layer_worker_slot_invalidated.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }

        perf().bytes_active_layer_upload_async.fetch_add((int64_t)copy_bytes, std::memory_order_relaxed);
        if (slot->gpu_id >= 0 && (size_t)slot->gpu_id < PerfCounters::PERF_MAX_GPU_SLOTS) {
            perf().bytes_h2d_upload_by_gpu[slot->gpu_id].fetch_add((int64_t)copy_bytes, std::memory_order_relaxed);
            perf().n_h2d_upload_by_gpu[slot->gpu_id].fetch_add(1, std::memory_order_relaxed);
        }

        // copy_dispatched flips true here, right after cudaEventRecord --
        // other threads can now start polling this slot's event via
        // find_active_expert/cleanup_active_layer_area. This is a
        // DESTINATION-side (VRAM) state transition only, unrelated to the
        // RAM-tier source's own protection below -- deferred to the
        // caller (worker_loop), batched with other already-completed
        // items into one cache_mutex acquisition instead of one per item.
        // Safe to defer: the copy itself is already fully confirmed done
        // (cudaStreamSynchronize above), so this is purely a visibility
        // bookkeeping step, not a completion gate -- a real dispatch that
        // reaches this slot in the small window before the batch flushes
        // just sees "not ready yet" (the existing, already-proven-safe
        // poll-and-fall-through-to-a-fresh-fetch path, not a correctness
        // issue) exactly as it already does for a slot whose DMA is
        // genuinely still in flight.
        //
        // RAM-tier source protection released via the same self-healing
        // pending_copy_event handoff decode_async_prewarm_nonblocking
        // already established -- no unregister step exists any more, so
        // there is nothing this ordering needs to happen before. Kept
        // per-item (not batched) -- unrelated lock (ram_mutex), no
        // measured contention there this session.
        ram_mgr_.ram_mark_pending_copy_and_unpin(*item.blk, slot->upload_event);

        perf().n_active_layer_worker_processed.fetch_add(1, std::memory_order_relaxed);
        return slot;
    }

    GpuTopology& topo_;
    RAMSafetyManager& ram_mgr_;
    size_t queue_capacity_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<ActiveLayerUploadWorkItem> queue_;
    bool shutting_down_ = false;
    std::vector<std::thread> workers_;
};
