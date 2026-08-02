#pragma once
#include "engine.hpp"
#include "gpu_topology.hpp"
#include "kquant_dequant.hpp"  // (kept: shared quant-type helpers used elsewhere in this header's includers)
#include "io_uring_backend.hpp"
#include "dense_dispatch.hpp"  // (no longer calls is_cpu_affinity_layer directly -- see finish_prefetch's comment on why the VRAM-promotion call, and its cpu_affinity gate, were removed; kept in case a future confirmed-use VRAM hook needs it again)
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <memory>
#include <string>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

// Any staging_mutex acquisition wait, or any staging-slot regrow, exceeding
// this is logged individually (tagged [SLOW-MUTEX-WAIT] / [SLOW-STAGING-
// REGROW]) -- added to root-cause a real, reproducible multi-second (up to
// 7.5s observed) single-miss stall that a per-miss latency histogram showed
// was NOT explained by raw disk throughput (p50/p90 were completely normal,
// only a handful of outlier calls were catastrophic). 50ms is already ~3x a
// normal ~15ms expert-slab read, so anything over this is unambiguously
// anomalous, not ordinary variance.
constexpr double SLOW_MUTEX_WAIT_MS = 50.0;

// How long a hint may sit in the queue before the worker thread actually
// starts processing it (pickup delay) before it's logged as anomalous
// ([SLOW-PICKUP]). Set higher than SLOW_MUTEX_WAIT_MS deliberately: even a
// perfectly healthy worker processes hints strictly sequentially, and the
// pilot pushes up to 8 hints per layer -- at a normal ~15ms/expert read,
// the LAST hint in one layer's batch can legitimately wait ~100ms+ just
// from the worker churning through the ones ahead of it, with zero
// contention or backlog involved. 250ms gives real margin above that
// benign case while still catching genuine backlog/starvation well before
// it reaches the multi-second severity seen in the SLOW-PREAD investigation.
// Added specifically to distinguish "the worker thread can't get scheduled
// promptly" (CPU contention with OMP dequant threads) from "the worker
// thread is scheduled fine but can't drain hints as fast as they're
// produced" (pure throughput/concurrency ceiling) -- the two point at very
// different fixes (nothing to do with thread count in the first case,
// more I/O parallelism in the second). NOTE: the "~100ms+ legitimate
// wait" baseline above assumed exactly ONE worker thread serializing all
// 8 hints; with prefetch_worker_threads > 1 (async_prefetcher's
// worker_loop, config prefetch_worker_threads) that worst case shrinks
// roughly proportionally, so this threshold is correspondingly more
// conservative (looser, not tighter) than strictly necessary when
// multiple workers are configured -- left as-is since a looser threshold
// only risks under-, not over-, reporting SLOW-PICKUP.
constexpr double SLOW_PICKUP_MS = 250.0;

struct PrefetchHint {
    int layer_idx;
    int expert_id;
    std::chrono::high_resolution_clock::time_point pushed_at;
};

// Staging slot: pinned host memory for an expert slab, consumed by
// RAMSafetyManager::get_experts_batch's consume_staged() to serve a genuine
// RAM-tier cache miss without a synchronous decode-thread pread. No longer
// carries a device_ptr/on_device pair -- finish_prefetch used to also fire
// an unconditional, unconfirmed VRAM promotion for every staged read (see
// its own comment for why that was removed); VRAM residency for a
// prefetched expert is now decided exclusively by the normal hit-count-
// gated promotion path once a real access confirms it's actually needed.
struct PrefetchSlot {
    uint8_t* pinned_buf = nullptr;  // cudaMallocHost'd buffer
    size_t capacity = 0;
    size_t loaded_bytes = 0;
    int layer_idx = -1;
    int expert_id = -1;
    std::atomic<bool> ready{false};      // true when pread is complete
};


class LockFreePrefetcher {
private:
    static constexpr size_t QUEUE_SIZE = 4096;
    // Was a compile-time constant (16, then bumped to 64 to test a specific
    // hypothesis: with pilot_k=8 hints pushed per layer transition and
    // multiple worker threads completing reads concurrently, a small
    // round-robin pool cycles fast enough that a correctly-predicted,
    // successfully-staged expert can get evicted before the corresponding
    // miss arrives -- i.e. pool THRASHING, not misprediction, as the
    // binding constraint on PIPE-served rate; confirmed by real-run
    // n_staged_evicted_unused counts). Now a runtime value (config
    // prefetch_staging_slots, engine_config.hpp), set once in start() before
    // `staging` is allocated and never changed after -- PrefetchSlot holds
    // an std::atomic<bool> member (ready) that is neither copyable
    // nor movable, so this can't be a std::vector (which needs move-safety
    // to resize/reallocate), but a fixed-size-at-allocation-time
    // std::unique_ptr<PrefetchSlot[]> has no such requirement.
    int num_staging_slots = 64;

    PrefetchHint queue[QUEUE_SIZE];
    std::atomic<unsigned int> head{0};
    std::atomic<unsigned int> tail{0};

    // Multiple worker threads pulling from the SAME queue -- see
    // worker_loop's CAS-based tail claim for why a plain load+store
    // advance (correct for exactly one consumer) is not safe once more
    // than one thread can race to claim the same slot.
    std::vector<std::thread> worker_threads;
    std::atomic<bool> running{false};
    StreamingEngine* engine = nullptr;
    // No longer read anywhere in this file (finish_prefetch's own comment
    // explains why the direct VRAM-promotion call was removed) -- kept as a
    // member/start() parameter only for call-site/API stability in case a
    // future, CONFIRMED-use-based VRAM hook is added here again.
    GpuTopology* gpu_topo = nullptr;

    // One io_uring ring PER WORKER THREAD -- a Ring is explicitly not
    // shareable across threads (io_uring_backend.hpp), so with N worker
    // threads each needs its own SQ/CQ. unique_ptr, not a plain Ring, so
    // growing this vector never needs Ring (or its AlignedScratch[]
    // member) to be copy/move-safe -- Ring owns a raw fd + several mmap'd
    // regions with no move constructor defined, and a vector<Ring> would
    // risk a double-close/double-munmap the moment resize() reallocates.
    // Sized to worker_threads.size() in start(); worker_loop(i) uses
    // *uring_rings[i] exclusively, so no locking is needed around any
    // individual ring. use_io_uring tracks whether io_uring was requested
    // AND at least one ring actually initialized (Linux + supported
    // kernel) -- purely for the startup log line; the per-ring
    // availability check inside try_uring_batch is what actually gates
    // behavior; a ring that individually failed to init just always takes
    // the sequential-pread fallback on its thread.
    std::vector<std::unique_ptr<io_uring_backend::Ring>> uring_rings;
    bool use_io_uring = false;

    // Pinned double-buffer pool for active prefetch. Allocated once in
    // start() (sized num_staging_slots) before any worker thread runs;
    // never resized afterward.
    std::unique_ptr<PrefetchSlot[]> staging;
    std::atomic<int> staging_write_idx{0};  // next slot to write into
    // Guards all mutation of a staging slot's identity (layer_idx/expert_id),
    // ready flag, AND the pinned_buf lifecycle (acquire_staging_slot
    // recycle/regrow vs consume_staged's memcpy-out). Without it the decode
    // thread's consume_staged could memcpy from a pinned_buf the worker thread
    // is concurrently cudaFreeHost-ing via acquire_staging_slot, and could read
    // a half-written (layer,expert) identity. finish_prefetch publishes
    // identity+ready under this lock; consume_staged + acquire_staging_slot take
    // it to read/recycle consistently.
    std::mutex staging_mutex;

public:
    void start(StreamingEngine* eng, GpuTopology* topo = nullptr, bool enable_io_uring = false,
               int n_worker_threads = 1, int n_staging_slots = 64) {
        engine = eng;
        gpu_topo = topo;
        // BUG FIX: this used to clamp UP to a minimum of 1, silently making
        // prefetch_worker_threads=0 impossible to actually test -- a real
        // investigation into whether the prefetcher's speculative reads
        // were competing with the decode thread's own critical-path reads
        // for scarce NVMe bandwidth (Pilot recall was a healthy 84%, but
        // Staged-unused evict showed ~98% of correctly-staged predictions
        // discarded before use) needed a genuine zero-worker baseline to
        // A/B against, and every attempt kept showing an active prefetcher
        // (Prefetch pickup: N hints processed, N>0) no matter what the
        // config said. Only clamp NEGATIVE values (a config typo) up to 0
        // -- 0 itself is now a real, honored "no prefetcher at all" state.
        // Every downstream use already tolerates zero cleanly:
        // uring_rings.resize(0) and worker_threads.reserve(0)/the spawn
        // loop below are no-ops, stop()/the destructor iterate empty
        // containers harmlessly, and callers already null-check-equivalent
        // (RAMSafetyManager::prefetcher stays a valid, non-null pointer to
        // this object either way, but consume_staged/has_pending_hint will
        // just always report nothing staged/pending -- correctness-neutral,
        // identical to how a normal cache miss with prefetching disabled
        // has always been handled).
        if (n_worker_threads < 0) n_worker_threads = 0;
        if (n_staging_slots < 1) n_staging_slots = 1;
        num_staging_slots = n_staging_slots;
        staging = std::make_unique<PrefetchSlot[]>(num_staging_slots);
        // Async H2D promotion goes through GpuTopology::upload_expert_async,
        // which uses the vault GPU's own dedicated copy_stream (see
        // gpu_topology.hpp) — no separate stream needed here.
        uring_rings.resize(n_worker_threads);
        for (auto& r : uring_rings) r = std::make_unique<io_uring_backend::Ring>();
        if (enable_io_uring) {
            int ok_count = 0;
            for (auto& r : uring_rings) {
                if (r->init()) {
                    // 3 = every active_prefetch() submission is exactly one
                    // hint's gate/up/down projections (io_uring_backend.hpp
                    // try_uring_batch calls), never more -- unlike the
                    // decode thread's own ring (see init_decode_uring),
                    // there's no larger batch size to size this pool for.
                    // Only meaningful once IOSQE_ASYNC actually forces
                    // these onto io-wq; see that flag's own comment.
                    r->register_iowq_workers(3);
                    ok_count++;
                }
            }
            use_io_uring = (ok_count > 0);
            if (use_io_uring) {
                LOG_STREAM_INFO() << "[IO-URING] Enabled on " << ok_count << "/" << n_worker_threads
                                   << " prefetch worker ring(s) -- batching per-expert reads through io_uring.\n";
            } else {
                LOG_STREAM_WARN() << "[IO-URING] io_uring=true requested but Ring::init() failed on all "
                                   << n_worker_threads << " worker ring(s) (old kernel, non-Linux, or "
                                      "sandboxed environment) -- falling back to sequential preads.\n";
            }
        }
        running.store(true, std::memory_order_release);
        worker_threads.reserve(n_worker_threads);
        for (int i = 0; i < n_worker_threads; ++i) {
            worker_threads.emplace_back(&LockFreePrefetcher::worker_loop, this, i);
        }
        LOG_STREAM_INFO() << "[I/O] Prefetcher: " << n_worker_threads << " worker thread(s).\n";
    }

    void stop() {
        running.store(false, std::memory_order_release);
        for (auto& t : worker_threads) {
            if (t.joinable()) t.join();
        }
    }

    ~LockFreePrefetcher() {
        if (!staging) return;  // start() never called (or failed before allocating)
        for (int i = 0; i < num_staging_slots; ++i) {
            if (staging[i].pinned_buf) cudaFreeHost(staging[i].pinned_buf);
        }
    }

    void push_hint(int layer_idx, int expert_id) {
        unsigned int current_head = head.load(std::memory_order_relaxed);
        unsigned int current_tail = tail.load(std::memory_order_acquire);

        if (current_head - current_tail >= QUEUE_SIZE) {
            return;
        }

        queue[current_head & (QUEUE_SIZE - 1)] = {layer_idx, expert_id, std::chrono::high_resolution_clock::now()};
        head.store(current_head + 1, std::memory_order_release);
    }

    // Diagnostic only (no locking, benign race with the workers' own
    // tail claims -- this is a "roughly right now" snapshot, not a
    // linearizable check): is there a hint for (layer_idx, expert_id)
    // still sitting in the queue, not yet claimed by any worker_loop
    // thread? Distinguishes, for a PIPE miss that found no staged slot,
    // "no worker has even started this read yet" (a scheduling/
    // throughput gap -- N hints queued per layer transition, serviced by
    // however many worker threads are configured) from "nothing was ever
    // predicted, or it was predicted+staged+already evicted by round-
    // robin before use" (a prediction-accuracy or pool-size gap).
    // Different root causes need different fixes; see
    // RAMSafetyManager::in_prefill's comment for the other half of this
    // same investigation.
    bool has_pending_hint(int layer_idx, int expert_id) const {
        unsigned int h = head.load(std::memory_order_acquire);
        unsigned int t = tail.load(std::memory_order_relaxed);
        for (unsigned int i = t; i != h; ++i) {
            const PrefetchHint& hint = queue[i & (QUEUE_SIZE - 1)];
            if (hint.layer_idx == layer_idx && hint.expert_id == expert_id) return true;
        }
        return false;
    }

    // Consume a prefetched slab: memcpy the staged bytes into the caller's
    // buffer under staging_mutex (so the worker can't recycle/free the pinned
    // slot mid-copy), then release the staging slot for reuse. Returns false
    // (no-op) if no ready staged slab matches this (layer, expert).
    //
    // This is the Phase A "PIPE" overlap path: a hit here serves a cache miss
    // WITHOUT a synchronous pread on the decode thread, because the prefetcher
    // worker already pread the bytes while a previous layer computed. The
    // previous check_staged() returned a raw pointer into the recycling pool
    // and was dead code anyway; this copy-out-under-lock is the safe, live
    // replacement. Caller is responsible for then setting up the per-projection
    // host_ptr/meta from `dst` (see StreamingEngine::populate_slab_from_buffer).
    bool consume_staged(int layer_idx, int expert_id, uint8_t* dst, size_t dst_cap, size_t* out_bytes) {
        if (!dst) return false;
        auto _wait_t0 = std::chrono::high_resolution_clock::now();
        std::lock_guard<std::mutex> lock(staging_mutex);
        double _wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - _wait_t0).count();
        PerfCounters::atomic_add(perf().t_staging_mutex_wait, _wait_ms / 1000.0);
        if (_wait_ms > SLOW_MUTEX_WAIT_MS) {
            LOG_WARN("[SLOW-MUTEX-WAIT] consume_staged blocked %.2fms acquiring staging_mutex "
                     "(layer=%d expert=%d) -- worker thread likely holding it, check for a nearby "
                     "[SLOW-STAGING-REGROW] line\n", _wait_ms, layer_idx, expert_id);
        }
        // Failure-reason split (n_pipe_miss_no_match / n_pipe_miss_not_ready,
        // perf_timer.hpp) -- added to diagnose a real run where Prefetch
        // (PIPE) served collapsed to 0.1%: matching on identity FIRST,
        // independent of ready, lets a caller distinguish "the pilot never
        // predicted this expert (or the slot got recycled before this miss
        // arrived)" from "the pilot predicted correctly but the read simply
        // hadn't finished yet" -- two different root causes needing two
        // different fixes. Behavior change from before: previously, a slot
        // with a matching identity but ready==false was silently skipped
        // and the scan continued (in case some OTHER slot also matched,
        // extremely unlikely in practice since round-robin write assignment
        // means at most one slot should ever hold a given identity at a
        // time); now the first identity match returns immediately either
        // way. Correctness-neutral either way -- ready==false still always
        // falls back to a synchronous load, exactly as before.
        for (int i = 0; i < num_staging_slots; ++i) {
            if (staging[i].layer_idx == layer_idx && staging[i].expert_id == expert_id) {
                if (!staging[i].ready.load(std::memory_order_acquire)) {
                    perf().n_pipe_miss_not_ready.fetch_add(1, std::memory_order_relaxed);
                    return false;  // predicted right, but the read isn't done -- caller falls back
                }
                size_t bytes = staging[i].loaded_bytes;
                if (dst_cap < bytes) return false;  // caller buffer too small -> caller falls back to load_expert_slab
                std::memcpy(dst, staging[i].pinned_buf, bytes);
                if (out_bytes) *out_bytes = bytes;
                // Release the staging slot for reuse. (We do NOT touch the VRAM
                // copy's completion -- that is tracked independently by
                // GpuTopology via the slot's upload_event in find_expert.)
                staging[i].ready.store(false, std::memory_order_release);
                staging[i].layer_idx = -1;
                staging[i].expert_id = -1;
                return true;
            }
        }
        perf().n_pipe_miss_no_match.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

private:
    // thread_idx selects this worker's own io_uring ring (uring_rings[thread_idx]).
    void worker_loop(int thread_idx) {
        io_uring_backend::Ring& my_ring = *uring_rings[thread_idx];
        while (running.load(std::memory_order_acquire) ||
               head.load(std::memory_order_acquire) != tail.load(std::memory_order_relaxed)) {

            unsigned int current_tail = tail.load(std::memory_order_relaxed);
            unsigned int current_head = head.load(std::memory_order_acquire);

            if (current_tail == current_head) {
                usleep(200);
                continue;
            }

            // Multiple worker threads share this queue -- CAS-claim a
            // unique slot instead of the old single-consumer load+store.
            // A plain store here would let a thread that read a stale
            // current_tail (e.g. preempted between the load above and
            // this point, while OTHER workers advanced tail further)
            // overwrite tail BACKWARD, causing hints to be reprocessed
            // and push_hint's overflow check to think the queue is
            // fuller than it really is. compare_exchange_weak only
            // succeeds if tail still equals what we just read; on
            // failure (another worker claimed this slot, or advanced
            // tail some other way) current_tail is updated to the real
            // value and we retry from the top with fresh head/tail reads.
            if (!tail.compare_exchange_weak(current_tail, current_tail + 1,
                                             std::memory_order_acq_rel, std::memory_order_relaxed)) {
                continue;
            }

            PrefetchHint hint = queue[current_tail & (QUEUE_SIZE - 1)];
            // Backlog depth AT the moment of pickup (includes the hint just
            // popped) -- a direct, independent signal for "is the worker
            // falling behind hint production", complementing the pickup-
            // delay measurement below (a growing backlog and a growing
            // delay should move together; if they don't, that itself is
            // informative -- e.g. a big backlog with small delay would mean
            // hints are being produced in a sudden burst but still drained
            // fast once picked up).
            unsigned int backlog = current_head - current_tail;

            double pickup_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - hint.pushed_at).count();
            PerfCounters::atomic_add(perf().t_prefetch_pickup_delay, pickup_ms / 1000.0);
            perf().n_prefetch_hints_processed.fetch_add(1, std::memory_order_relaxed);
            PerfCounters::atomic_max(perf().n_prefetch_backlog_peak, (uint64_t)backlog);
            if (pickup_ms > SLOW_PICKUP_MS) {
                LOG_WARN("[SLOW-PICKUP] layer=%d expert=%d waited %.2fms in queue before the worker "
                         "thread picked it up (queue backlog at pickup: %u hints) -- worker is falling "
                         "behind hint production, either CPU-starved or I/O-throughput-bound\n",
                         hint.layer_idx, hint.expert_id, pickup_ms, backlog);
            }

            active_prefetch(hint.layer_idx, hint.expert_id, my_ring);
        }
    }

    // Tries to service up to 3 reads (gate/up/down) as ONE io_uring batch
    // instead of sequential preads. All-or-nothing: returns false (no
    // partial application, no bytes written anywhere) unless io_uring is
    // enabled+available AND every single request came back with exactly
    // its requested length -- any short read/error on any leg falls
    // through to the caller's existing sequential pread_direct_aware path,
    // which simply redoes all `n` reads. Prefetch failures are rare enough
    // that the occasional wasted re-read costs nothing next to keeping
    // this a single unconditional correctness contract.
    //
    // Records n_uring_batch_ok/n_uring_batch_fallback ONLY for genuine
    // attempts (io_uring enabled AND the ring initialized) -- see those
    // counters' own comment in perf_timer.hpp for why: the fallback path
    // here is silent by design, so without this counter there's no way to
    // tell "io_uring is on but every batch is failing" from "io_uring is
    // helping" purely from a run's throughput numbers.
    bool try_uring_batch(const io_uring_backend::ReadRequest* reqs, int n, io_uring_backend::Ring& my_ring) {
        if (!my_ring.available()) return false;
        ssize_t results[3];
        bool ok = my_ring.submit_and_wait(reqs, n, results);
        if (ok) {
            for (int i = 0; i < n; ++i) {
                if (results[i] != (ssize_t)reqs[i].len) { ok = false; break; }
            }
        }
        if (ok) perf().n_uring_batch_ok.fetch_add(1, std::memory_order_relaxed);
        else perf().n_uring_batch_fallback.fetch_add(1, std::memory_order_relaxed);
        return ok;
    }

    // Active prefetch: pread into pinned host memory (not just fadvise), then
    // (if a VRAM vault is configured) hand the pinned buffer to
    // GpuTopology::upload_expert_async for the actual H2D copy — that
    // function owns the slot-claim + stamp + memcpy sequence under
    // cache_mutex, so this never races the main thread's own promotions.
    // my_ring is the CALLING worker thread's own io_uring ring (see
    // worker_loop) -- never shared across threads.
    void active_prefetch(int layer_idx, int expert_id, io_uring_backend::Ring& my_ring) {
        if (!engine) return;

        // Check if already staged (under the same lock that publish/consume
        // use, so we never read a half-written identity).
        {
            auto _wait_t0 = std::chrono::high_resolution_clock::now();
            std::lock_guard<std::mutex> lock(staging_mutex);
            double _wait_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - _wait_t0).count();
            PerfCounters::atomic_add(perf().t_staging_mutex_wait, _wait_ms / 1000.0);
            if (_wait_ms > SLOW_MUTEX_WAIT_MS) {
                LOG_WARN("[SLOW-MUTEX-WAIT] active_prefetch blocked %.2fms acquiring staging_mutex "
                         "(layer=%d expert=%d)\n", _wait_ms, layer_idx, expert_id);
            }
            for (int i = 0; i < num_staging_slots; ++i) {
                if (staging[i].layer_idx == layer_idx &&
                    staging[i].expert_id == expert_id) return; // already in flight
            }
        }

        std::string base = "blk." + std::to_string(layer_idx) + ".";
        int n_exp = engine->model_file.architecture_config.n_experts;

        // ---- Packed format: ffn_{gate,up,down}_exps.weight. Expert E's slice
        // is [E*per_exp, (E+1)*per_exp) within each tensor — mirrors
        // StreamingEngine::load_expert_slab's packed branch exactly, including
        // per-projection quant-type sizing (GLM-5.2 UD: gate/up=Q2_K,
        // down=Q3_K, NOT uniformly sized). The old code here only read
        // `size_bytes*3/n_experts` bytes starting at the GATE tensor's base
        // offset (expert 0's slice, regardless of `expert_id`) and called that
        // "the expert" — silently mislabeling every promoted slot.
        auto gE = engine->model_file.tensors.find(base + "ffn_gate_exps.weight");
        auto uE = engine->model_file.tensors.find(base + "ffn_up_exps.weight");
        auto dE = engine->model_file.tensors.find(base + "ffn_down_exps.weight");
        if (n_exp > 0 && gE != engine->model_file.tensors.end() &&
            uE != engine->model_file.tensors.end() && dE != engine->model_file.tensors.end()) {
            // NOTE: this used to also skip here whenever gpu_topo was set
            // but this layer's quant types couldn't dispatch to GPU --
            // reasoning at the time was "this whole prefetch cycle exists
            // only to feed the async VRAM promotion below." That VRAM
            // promotion has since been REMOVED from finish_prefetch (see
            // its own comment): every genuine consumer of a staged read is
            // now RAMSafetyManager::get_experts_batch's consume_staged
            // call, which benefits a RAM-tier miss regardless of whether
            // the expert is GPU-dispatchable or CPU-fallback-dispatched --
            // so skipping GPU-incapable experts here would wrongly disable
            // prefetch for exactly the (CPU-fallback) experts that still
            // pay a real synchronous disk-read cost on the decode thread.
            const size_t gb = gE->second.size_bytes / n_exp;
            const size_t ub = uE->second.size_bytes / n_exp;
            const size_t db = dE->second.size_bytes / n_exp;
            const size_t total = gb + ub + db;

            PrefetchSlot& slot = acquire_staging_slot(total);
            if (!slot.pinned_buf) { issue_fadvise_fallback(layer_idx, expert_id); return; }

            int gfd = engine->model_file.fd_for_tensor(gE->second);
            int ufd = engine->model_file.fd_for_tensor(uE->second);
            int dfd = engine->model_file.fd_for_tensor(dE->second);
            uint64_t align = engine->model_file.direct_io_alignment;

            // All three projections are scattered (packed-format experts
            // are 3 separate on-disk slices by construction) -- exactly the
            // case io_uring's batching helps: submit all 3 as one call
            // instead of 3 sequential blocking preads. Falls through to the
            // existing sequential path on any failure.
            io_uring_backend::ReadRequest reqs[3] = {
                {gfd, engine->model_file.direct_io_active_for(gE->second), align,
                 slot.pinned_buf, gb, gE->second.offset + (size_t)expert_id * gb},
                {ufd, engine->model_file.direct_io_active_for(uE->second), align,
                 slot.pinned_buf + gb, ub, uE->second.offset + (size_t)expert_id * ub},
                {dfd, engine->model_file.direct_io_active_for(dE->second), align,
                 slot.pinned_buf + gb + ub, db, dE->second.offset + (size_t)expert_id * db},
            };
            bool ok = try_uring_batch(reqs, 3, my_ring);
            if (!ok) {
                ssize_t rg = direct_io::pread_direct_aware(
                    gfd, engine->model_file.direct_io_active_for(gE->second), align,
                    slot.pinned_buf, gb, gE->second.offset + (size_t)expert_id * gb);
                ssize_t ru = direct_io::pread_direct_aware(
                    ufd, engine->model_file.direct_io_active_for(uE->second), align,
                    slot.pinned_buf + gb, ub, uE->second.offset + (size_t)expert_id * ub);
                ssize_t rd = direct_io::pread_direct_aware(
                    dfd, engine->model_file.direct_io_active_for(dE->second), align,
                    slot.pinned_buf + gb + ub, db, dE->second.offset + (size_t)expert_id * db);
                ok = (rg == (ssize_t)gb && ru == (ssize_t)ub && rd == (ssize_t)db);
            }

            if (ok) {
                finish_prefetch(slot, layer_idx, expert_id, total);
            } else {
                issue_fadvise_fallback(layer_idx, expert_id);
            }
            return;
        }

        // ---- Per-expert format: ffn_{gate,up,down}.E.weight ----
        // Read each projection INDEPENDENTLY into a TIGHT gate|up|down staging
        // buffer (gate@0, up@gb, down@gb+ub), mirroring StreamingEngine::
        // load_expert_slab's per-expert path EXACTLY so the bytes uploaded to
        // VRAM here match the slab-relative meta deltas the GPU dispatch
        // indexes by (expert_ffn_gpu_issue: up@gb, down@gb+ub). Layout-agnostic:
        // no assumption about the relative file offsets. The old single-combined
        // pread over [gate.offset, down.end) underflowed size_t (down<gate<up
        // after a lexicographic repack sort) -> acquire_staging_slot(SIZE_MAX)
        // -> cudaMallocHost failed -> silent fadvise fallback, so per-expert
        // experts were never promoted to VRAM at all.
        std::string gate_name = base + "ffn_gate." + std::to_string(expert_id) + ".weight";
        std::string up_name   = base + "ffn_up."   + std::to_string(expert_id) + ".weight";
        std::string down_name = base + "ffn_down." + std::to_string(expert_id) + ".weight";
        auto git = engine->model_file.tensors.find(gate_name);
        auto dit = engine->model_file.tensors.find(down_name);
        if (git == engine->model_file.tensors.end() || dit == engine->model_file.tensors.end()) {
            issue_fadvise_fallback(layer_idx, expert_id);
            return;
        }
        const GGUFTensor& gate = git->second;
        const GGUFTensor& down = dit->second;
        auto uit = engine->model_file.tensors.find(up_name);
        const GGUFTensor* up = (uit != engine->model_file.tensors.end()) ? &uit->second : nullptr;

        const size_t gb = gate.size_bytes;
        const size_t ub = up ? up->size_bytes : 0;
        const size_t db = down.size_bytes;
        const size_t total = gb + ub + db;

        PrefetchSlot& slot = acquire_staging_slot(total);
        if (!slot.pinned_buf) { issue_fadvise_fallback(layer_idx, expert_id); return; }

        uint64_t align = engine->model_file.direct_io_alignment;
        // Single-read fast path (mirrors load_expert_slab): if the three
        // projections are packed tight in one same-shard expert block, fetch the whole expert with ONE
        // pread into the tight staging buffer instead of three.
        int gate_fd = engine->model_file.fd_for_tensor(gate);
        bool tight = up
            && engine->model_file.fd_for_tensor(*up) == gate_fd
            && engine->model_file.fd_for_tensor(down) == gate_fd
            && up->offset == gate.offset + gb
            && down.offset == up->offset + ub;
        bool ok;
        if (tight) {
            ssize_t r = direct_io::pread_direct_aware(
                gate_fd, engine->model_file.direct_io_active_for(gate), align,
                slot.pinned_buf, total, gate.offset);
            ok = (r == (ssize_t)total);
        } else {
            // Scattered projections (not same-shard/tight) -- same
            // io_uring batching opportunity as the packed-format branch
            // above. `up` can be absent (ub=0), so the batch is 2 or 3
            // requests depending on the model.
            io_uring_backend::ReadRequest reqs[3];
            int n = 0;
            reqs[n++] = {engine->model_file.fd_for_tensor(gate), engine->model_file.direct_io_active_for(gate),
                         align, slot.pinned_buf, gb, gate.offset};
            if (up) {
                reqs[n++] = {engine->model_file.fd_for_tensor(*up), engine->model_file.direct_io_active_for(*up),
                             align, slot.pinned_buf + gb, ub, up->offset};
            }
            reqs[n++] = {engine->model_file.fd_for_tensor(down), engine->model_file.direct_io_active_for(down),
                         align, slot.pinned_buf + gb + ub, db, down.offset};
            ok = try_uring_batch(reqs, n, my_ring);
            if (!ok) {
                ssize_t rg = direct_io::pread_direct_aware(
                    engine->model_file.fd_for_tensor(gate), engine->model_file.direct_io_active_for(gate), align,
                    slot.pinned_buf, gb, gate.offset);
                ssize_t ru = 0;
                if (up) ru = direct_io::pread_direct_aware(
                    engine->model_file.fd_for_tensor(*up), engine->model_file.direct_io_active_for(*up), align,
                    slot.pinned_buf + gb, ub, up->offset);
                ssize_t rd = direct_io::pread_direct_aware(
                    engine->model_file.fd_for_tensor(down), engine->model_file.direct_io_active_for(down), align,
                    slot.pinned_buf + gb + ub, db, down.offset);
                ok = (rg == (ssize_t)gb && (!up || ru == (ssize_t)ub) && rd == (ssize_t)db);
            }
        }

        if (ok) {
            finish_prefetch(slot, layer_idx, expert_id, total);
        } else {
            issue_fadvise_fallback(layer_idx, expert_id);
        }
    }

    // Round-robin staging slot, grown to `needed_bytes` if too small. Takes
    // staging_mutex so the regrow (which can cudaFreeHost the pinned_buf) can't
    // race a concurrent consume_staged memcpy-out of the same slot.
    PrefetchSlot& acquire_staging_slot(size_t needed_bytes) {
        int slot_idx = staging_write_idx.fetch_add(1, std::memory_order_relaxed) % num_staging_slots;
        auto _wait_t0 = std::chrono::high_resolution_clock::now();
        std::lock_guard<std::mutex> lock(staging_mutex);
        double _wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - _wait_t0).count();
        PerfCounters::atomic_add(perf().t_staging_mutex_wait, _wait_ms / 1000.0);
        if (_wait_ms > SLOW_MUTEX_WAIT_MS) {
            LOG_WARN("[SLOW-MUTEX-WAIT] acquire_staging_slot blocked %.2fms acquiring staging_mutex (slot=%d)\n",
                     _wait_ms, slot_idx);
        }
        PrefetchSlot& slot = staging[slot_idx];
        // Diagnostic: are we about to overwrite a slot that was
        // successfully staged (ready==true) but NEVER consumed? A
        // successful consume_staged() resets layer_idx/expert_id to -1 on
        // its way out (see its own body below), so ready==true with a
        // still-real (layer,expert) identity here means this exact
        // prediction was correctly made, correctly read, correctly
        // staged -- and simply never matched a miss before round-robin
        // rotation came back around to recycle its slot. Distinguishes
        // "pool thrashing" from other explanations for the gap between
        // Pilot recall (prediction accuracy alone) and Prefetch (PIPE)
        // served % (the full pipeline) -- safe to read here since we
        // already hold staging_mutex, the same lock consume_staged uses
        // to mutate these same fields.
        if (slot.ready.load(std::memory_order_relaxed) && slot.layer_idx != -1) {
            perf().n_staged_evicted_unused.fetch_add(1, std::memory_order_relaxed);
        }
        if (slot.capacity < needed_bytes) {
            // PRIME SUSPECT for the multi-second single-miss stall: pinned
            // (page-locked) host memory alloc/free is a documented-slow CUDA
            // driver operation (can run tens to thousands of ms, especially
            // under memory pressure or fragmentation) -- and this whole pair
            // runs WHILE HOLDING staging_mutex, so any concurrent
            // consume_staged() call on the decode thread blocks for the full
            // duration. A too-small pool (num_staging_slots, config
            // prefetch_staging_slots) rotating round-robin across MoE layers
            // whose packed slab size can differ layer-to-layer (different
            // quant types e.g. IQ4_XS vs IQ3_XXS down-projections) means this
            // regrow can trigger far more often than "only once at startup"
            // -- every time the round-robin lands on a slot last sized for a
            // DIFFERENT layer's (smaller) slab.
            auto _regrow_t0 = std::chrono::high_resolution_clock::now();
            if (slot.pinned_buf) cudaFreeHost(slot.pinned_buf);
            cudaMallocHost((void**)&slot.pinned_buf, needed_bytes);
            double _regrow_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - _regrow_t0).count();
            PerfCounters::atomic_add(perf().t_staging_regrow, _regrow_ms / 1000.0);
            perf().n_staging_regrow.fetch_add(1, std::memory_order_relaxed);
            if (_regrow_ms > SLOW_MUTEX_WAIT_MS) {
                LOG_WARN("[SLOW-STAGING-REGROW] slot=%d cudaFreeHost+cudaMallocHost took %.2fms "
                         "(old_capacity=%.2fMB new=%.2fMB) -- held staging_mutex for the whole "
                         "duration, blocking any concurrent consume_staged\n",
                         slot_idx, slot.capacity / 1e6, needed_bytes / 1e6, _regrow_ms);
            }
            slot.capacity = needed_bytes;
        }
        return slot;
    }

    // Publish identity+ready under staging_mutex.
    //
    // REAL BUG FOUND AND FIXED (root cause of near-total PIPE waste): this
    // used to ALSO fire an unconditional, unconfirmed speculative
    // gpu_topo->upload_expert_async(...) promotion here, on EVERY
    // successfully-staged read, before publishing identity. Two compounding
    // problems, found by reading this function together with
    // execution_layer.cpp's shadow pre-check and RAMSafetyManager::
    // get_experts_batch's consume_staged call:
    //
    // 1) upload_expert_async claims a REAL VRAM cache slot via evict_lru --
    //    i.e. every prefetch read, correct prediction or not, evicted an
    //    ALREADY-VRAM-RESIDENT (possibly genuinely useful) expert to make
    //    room for a merely-speculative one. A wrong prediction (Pilot
    //    recall was only 84%, so ~16% of pushed hints are already wrong on
    //    prediction accuracy alone) didn't just waste a staging slot -- it
    //    also evicted something real for nothing. This is a second,
    //    independent, previously-undocumented waste mechanism on top of the
    //    RAM-staging-pool churn CLAUDE.md's history already covered.
    // 2) When the speculative promotion DID land (async H2D completed)
    //    before decode reached this layer, execution_layer.cpp's shadow
    //    pre-check (find_expert) found the expert ALREADY VRAM-resident and
    //    dispatched it via the shadow fast path -- which never calls
    //    get_experts_batch/consume_staged at all for a shadowed expert. So
    //    the RAM pinned buffer this whole prefetch cycle just filled was
    //    never read by anyone: not a timing loss, not a misprediction, but
    //    the SAME staged data being "used" via a different path than the
    //    one the Staged-unused-evict counter was measuring, which is why a
    //    healthy-looking Pilot recall (84%) coexisted with ~98.5%
    //    Staged-unused-evict and ~0.1% PIPE-served in every real run this
    //    was measured on. The unconfirmed-promotion side effect is also the
    //    documented root cause behind gpu_topology.hpp's own
    //    n_stale_upload_ready_reclaimed counter (37.7% of VRAM slots found
    //    permanently stuck mid-upload in one real run) -- that fix only
    //    mitigated the symptom (evict_lru self-healing a stale flag), it
    //    never removed the unconfirmed speculative promotion causing it.
    //
    // Fix: this function now ONLY stages into the pinned RAM buffer and
    // feeds consume_staged -- it no longer touches VRAM at all. VRAM
    // promotion for a prefetched expert now happens exactly the same way
    // as for any other RAM-tier resident: via the existing hit-count-gated
    // path in get_expert/get_experts_batch (promote_hit_threshold), which
    // only fires once a real subsequent access CONFIRMS the expert is
    // actually being used -- never on unconfirmed speculation. This also
    // means the shadow pre-check can no longer be prematurely satisfied by
    // a prefetch that hasn't been consumed yet, so a genuine RAM-tier miss
    // for a correctly-predicted expert now actually reaches consume_staged
    // and gets served from it, instead of being raced out from under it.
    void finish_prefetch(PrefetchSlot& slot, int layer_idx, int expert_id, size_t total_bytes) {
        slot.loaded_bytes = total_bytes;

        auto _wait_t0 = std::chrono::high_resolution_clock::now();
        std::lock_guard<std::mutex> lock(staging_mutex);
        double _wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - _wait_t0).count();
        PerfCounters::atomic_add(perf().t_staging_mutex_wait, _wait_ms / 1000.0);
        if (_wait_ms > SLOW_MUTEX_WAIT_MS) {
            LOG_WARN("[SLOW-MUTEX-WAIT] finish_prefetch blocked %.2fms acquiring staging_mutex "
                     "(layer=%d expert=%d)\n", _wait_ms, layer_idx, expert_id);
        }
        slot.layer_idx = layer_idx;
        slot.expert_id = expert_id;
        slot.ready.store(true, std::memory_order_release);
    }

    void issue_fadvise_fallback(int layer_idx, int expert_id) {
        // Prefetch each projection by name. The old (gate.offset, gate.size*3)
        // hint assumed up/down follow gate contiguously AND share its size --
        // neither holds for a repacked (down<gate<up) or mixed-quant layout.
        std::string base = "blk." + std::to_string(layer_idx) + ".";
        std::string es = std::to_string(expert_id);
        for (const char* proj : {"ffn_gate", "ffn_up", "ffn_down"}) {
            auto it = engine->model_file.tensors.find(base + proj + "." + es + ".weight");
            if (it != engine->model_file.tensors.end()) {
                posix_fadvise(engine->model_file.fd_for_tensor(it->second),
                              it->second.offset, it->second.size_bytes, POSIX_FADV_WILLNEED);
            }
        }
    }
};