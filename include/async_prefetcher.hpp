#pragma once
#include "engine.hpp"
#include "gpu_topology.hpp"
#include "ram_manager.hpp"  // RAMSafetyManager::claim_or_wait -- see active_prefetch's own comment for the unified claim/publish design this now goes through. Safe to include directly (no cycle): ram_manager.hpp no longer includes this header back (see RAMSafetyManager::ram_mutex's own comment for why the old prefetcher-pointer field was removed).
#include "kquant_dequant.hpp"  // (kept: shared quant-type helpers used elsewhere in this header's includers)
#include "io_uring_backend.hpp"
#include "expert_fetch_tracker.hpp"  // per-(layer,expert) fetch-state record -- kept for the VRAM-tier shadow-fast-path signal only (execution_layer.cpp); the RAM-tier race it used to detect is now structurally impossible (see claim_or_wait's own comment), so active_prefetch below only ever marks InFlight
#include <atomic>
#include <thread>
#include <vector>
#include <memory>
#include <string>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

// How long a hint may sit in the queue before the worker thread actually
// starts processing it (pickup delay) before it's logged as anomalous
// ([SLOW-PICKUP]). Even a perfectly healthy worker processes hints strictly
// sequentially, and the pilot pushes up to 8 hints per layer -- at a normal
// ~15ms/expert read, the LAST hint in one layer's batch can legitimately
// wait ~100ms+ just from the worker churning through the ones ahead of it,
// with zero contention or backlog involved. 250ms gives real margin above
// that benign case while still catching genuine backlog/starvation.
// Distinguishes "the worker thread can't get scheduled promptly" (CPU
// contention with OMP dequant threads) from "the worker thread is scheduled
// fine but can't drain hints as fast as they're produced" (pure throughput/
// concurrency ceiling) -- the two point at very different fixes. NOTE: the
// "~100ms+ legitimate wait" baseline above assumed exactly ONE worker thread
// serializing all 8 hints; with prefetch_worker_threads > 1 that worst case
// shrinks roughly proportionally, so this threshold is correspondingly more
// conservative (looser, not tighter) than strictly necessary when multiple
// workers are configured -- left as-is since a looser threshold only risks
// under-, not over-, reporting SLOW-PICKUP.
constexpr double SLOW_PICKUP_MS = 250.0;

// Which pilot fired this hint -- exists purely to split the shared queue's
// real disk-cost counters (bytes/time/issued/completed/failed/claims_
// beaten, perf_timer.hpp) by source, so DEPTH-2's actual marginal disk-I/O
// cost (EngineConfig::prefetch_pilot_depth=2, execution_layer.cpp's l+2
// block) is a measured number instead of an estimate extrapolated from
// recall stats alone. Depth2 = l+2 pilot hints. Other = everything else
// that shares this same queue (l+1's own pilot hints, and Phase 2's
// prefill_warm_via_prefetcher hints) -- NOT split further, since isolating
// depth-2's cost is this split's only purpose; "Other" is not a synonym
// for "depth-1 pilot only".
enum class PilotSource : uint8_t { Other, Depth2 };

struct PrefetchHint {
    int layer_idx;
    int expert_id;
    std::chrono::high_resolution_clock::time_point pushed_at;
    PilotSource source = PilotSource::Other;
};

// Per-layer prefetch activity, sized to the model's layer count once in
// LockFreePrefetcher::start() and never resized after -- the process-wide
// totals in perf_timer.hpp can't tell you WHICH layers the prefetcher
// actually helps vs wastes bandwidth on; this answers that per-layer via
// LockFreePrefetcher::log_per_layer_report(). Rewritten for the unified
// claim/publish design (RAMSafetyManager::claim_or_wait): the old prefetch_
// served/prefetch_no_match/prefetch_not_ready/staged_evicted_unused/used_lifetime_
// us_sum/evicted_lifetime_us_sum fields described a separate staging pool
// that no longer exists -- claims_beaten_by_decode is the new, directly
// analogous signal (a prefetch worker's OWN claim attempt for this layer
// found the expert already Hit or Loading -- i.e. the decode thread's own
// miss-handling, or another prefetch worker, won the claim first).
struct LayerPrefetchStats {
    std::atomic<uint64_t> hints_pushed{0};           // push_hint calls for this layer
    std::atomic<uint64_t> fetches_issued{0};         // real disk reads actually started (active_prefetch)
    std::atomic<uint64_t> fetches_completed{0};      // ...and finished successfully
    std::atomic<uint64_t> fetches_failed{0};         // ...and finished with a short-read/error
    std::atomic<uint64_t> claims_beaten_by_decode{0}; // this layer's claim_or_wait attempt found the expert already resident or already being fetched by someone else -- a wasted hint, now directly countable
    std::atomic<uint64_t> fetch_us_sum{0};           // sum of completed-fetch durations (microseconds) -- for this layer's avg_ms
};


class LockFreePrefetcher {
private:
    static constexpr size_t QUEUE_SIZE = 4096;

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

    // Per-layer prefetch activity (see LayerPrefetchStats's own comment).
    // Allocated once in start() from the model's layer count, never
    // resized -- layer_stat() bounds-checks every access so an
    // out-of-range layer_idx (shouldn't happen) is silently ignored
    // rather than crashing.
    std::unique_ptr<LayerPrefetchStats[]> layer_stats;
    int n_layers_tracked = 0;
    LayerPrefetchStats* layer_stat(int layer_idx) {
        return (layer_stats && layer_idx >= 0 && layer_idx < n_layers_tracked)
            ? &layer_stats[layer_idx] : nullptr;
    }

public:
    // Live gauge (not a perf() cumulative counter): how many real fetches
    // are currently between record_fetch_start and record_fetch_end. A
    // cheap, always-current number, no scan needed. Public (unlike most
    // internal state here) precisely because an external read is the point
    // of tracking it.
    std::atomic<int> n_fetches_in_flight{0};

    // Unified RAM tier this prefetcher claims/populates/publishes through
    // (RAMSafetyManager::claim_or_wait) -- set post-construction in main.cpp
    // (mirrors RAMSafetyManager's own pre-existing raw-pointer-set-after-
    // construction pattern for vram_cache). Raw, non-owning: RAMSafetyManager
    // outlives this prefetcher for the whole process lifetime. Public
    // (mirroring RAMSafetyManager's own public fields) since main.cpp must
    // set it directly from outside this class.
    RAMSafetyManager* ram_mgr = nullptr;

    // avg ms per completed real fetch so far this run.
    // t_prefetch_read_ms_sum is a sum across possibly-concurrent worker
    // threads (see its own comment in perf_timer.hpp) but dividing by a
    // count of completed fetches still yields a valid mean duration per
    // fetch -- it's only invalid to read the sum as a wall-clock total.
    double avg_fetch_ms() const {
        uint64_t completed = perf().n_prefetch_completed.load(std::memory_order_relaxed);
        return completed > 0 ? perf().t_prefetch_read_ms_sum.load(std::memory_order_relaxed) / (double)completed : 0.0;
    }

    // Per-layer prefetch report, printed at shutdown (main.cpp, after
    // prefetcher.stop()) alongside the rest of the perf report. The
    // process-wide totals in perf_timer.hpp can't answer "which LAYERS is
    // the prefetcher actually helping vs wasting bandwidth on" -- this does,
    // one line per layer that ever saw any prefetch activity (a layer that
    // never touched the prefetcher at all -- e.g. cpu_affinity_layers -- is
    // skipped rather than padding the log with an all-zero row). Rewritten
    // for the unified claim/publish design: "beaten" replaces the old pipe_*
    // vocabulary (see LayerPrefetchStats's own comment) -- a claim attempt
    // that found the expert already resident or already being fetched by
    // someone else is a wasted hint, now directly countable via claim_or_
    // wait's own return value instead of inferred from a separate staging
    // pool's own bookkeeping. See perf_timer.hpp's "Unified RAM claim"/
    // "Unified RAM waste" report lines for the process-wide counterparts
    // (n_ram_claim_already_resident_prefetch/already_inflight_prefetch,
    // n_ram_prefetch_loaded_evicted_unused).
    void log_per_layer_report() const {
        if (!layer_stats || n_layers_tracked == 0) return;
        LOG_STREAM_INFO() << "[PREFETCH-LAYER] per-layer prefetch activity (of " << n_layers_tracked
                           << " layer(s); hints=pushed to queue, issued/done/fail=real disk reads "
                              "attempted/completed/failed, beaten=this layer's claim attempt found the "
                              "expert already resident or already being fetched by someone else (a wasted "
                              "hint), avg_ms=mean completed-fetch duration):\n";
        uint64_t total_issued = 0, total_beaten = 0;
        for (int l = 0; l < n_layers_tracked; ++l) {
            const LayerPrefetchStats& s = layer_stats[l];
            uint64_t hints = s.hints_pushed.load(std::memory_order_relaxed);
            uint64_t issued = s.fetches_issued.load(std::memory_order_relaxed);
            uint64_t beaten = s.claims_beaten_by_decode.load(std::memory_order_relaxed);
            if (hints == 0 && issued == 0 && beaten == 0) continue;
            uint64_t completed = s.fetches_completed.load(std::memory_order_relaxed);
            uint64_t failed = s.fetches_failed.load(std::memory_order_relaxed);
            double avg_ms = completed > 0
                ? (double)s.fetch_us_sum.load(std::memory_order_relaxed) / 1000.0 / (double)completed : 0.0;
            total_issued += issued;
            total_beaten += beaten;
            LOG_STREAM_INFO() << "[PREFETCH-LAYER]   layer=" << l << "  hints=" << hints
                               << " issued=" << issued << " done=" << completed << " fail=" << failed
                               << " beaten=" << beaten << " avg_ms=" << avg_ms << "\n";
        }
        uint64_t total_attempts = total_issued + total_beaten;
        LOG_STREAM_INFO() << "[PREFETCH-LAYER] claim outcome: of " << total_attempts << " claim attempt(s), "
                           << total_issued << " ("
                           << (total_attempts > 0 ? 100.0 * (double)total_issued / (double)total_attempts : 0.0)
                           << "%) won the claim and issued a real fetch, " << total_beaten << " ("
                           << (total_attempts > 0 ? 100.0 * (double)total_beaten / (double)total_attempts : 0.0)
                           << "%) were beaten to it (already resident or already being fetched by the "
                              "decode thread or another prefetch worker) -- structurally impossible to "
                              "duplicate a fetch under the unified claim design, see claim_or_wait's own "
                              "comment (ram_manager.hpp)\n";
    }

    void start(StreamingEngine* eng, GpuTopology* topo = nullptr, bool enable_io_uring = false,
               int n_worker_threads = 1) {
        engine = eng;
        gpu_topo = topo;
        // BUG FIX: this used to clamp UP to a minimum of 1, silently making
        // prefetch_worker_threads=0 impossible to actually test -- a real
        // investigation into whether the prefetcher's speculative reads
        // were competing with the decode thread's own critical-path reads
        // for scarce NVMe bandwidth needed a genuine zero-worker baseline to
        // A/B against, and every attempt kept showing an active prefetcher
        // (Prefetch pickup: N hints processed, N>0) no matter what the
        // config said. Only clamp NEGATIVE values (a config typo) up to 0
        // -- 0 itself is now a real, honored "no prefetcher at all" state.
        // Every downstream use already tolerates zero cleanly:
        // uring_rings.resize(0) and worker_threads.reserve(0)/the spawn
        // loop below are no-ops, stop() iterates an empty container
        // harmlessly, and ram_mgr's own claim_or_wait works identically
        // whether or not a prefetcher is running (correctness-neutral,
        // identical to how a normal cache miss with prefetching disabled
        // has always been handled).
        if (n_worker_threads < 0) n_worker_threads = 0;
        n_layers_tracked = eng ? eng->model_file.architecture_config.n_layers : 0;
        if (n_layers_tracked > 0) layer_stats = std::make_unique<LayerPrefetchStats[]>(n_layers_tracked);
        if (n_layers_tracked > 0 && eng) {
            expert_fetch_tracker().init(n_layers_tracked, eng->model_file.architecture_config.n_experts);
        }
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

    void push_hint(int layer_idx, int expert_id, PilotSource source = PilotSource::Other) {
        unsigned int current_head = head.load(std::memory_order_relaxed);
        unsigned int current_tail = tail.load(std::memory_order_acquire);

        if (current_head - current_tail >= QUEUE_SIZE) {
            return;
        }

        queue[current_head & (QUEUE_SIZE - 1)] = {layer_idx, expert_id, std::chrono::high_resolution_clock::now(), source};
        head.store(current_head + 1, std::memory_order_release);
        if (auto* s = layer_stat(layer_idx)) s->hints_pushed.fetch_add(1, std::memory_order_relaxed);
    }

    // Diagnostic only (no locking, benign race with the workers' own
    // tail claims -- this is a "roughly right now" snapshot, not a
    // linearizable check): is there a hint for (layer_idx, expert_id)
    // still sitting in the queue, not yet claimed by any worker_loop
    // thread? Distinguishes, for a prefetch miss that found no staged slot,
    // "no worker has even started this read yet" (a scheduling/
    // throughput gap -- N hints queued per layer transition, serviced by
    // however many worker threads are configured) from "nothing was ever
    // predicted, or it was predicted+staged+already evicted by round-
    // robin before use" (a prediction-accuracy or pool-size gap).
    // Different root causes need different fixes; see
    // RAMSafetyManager::in_prefill's comment for the other half of this
    // same investigation.
    // out_queued_ms (optional): when a match is found, populated with how
    // long (ms) the hint has been sitting in the queue since push_hint --
    // i.e. how much of a head start the prefetcher had before the decode
    // thread needed this expert, versus how much it actually got (zero
    // worker attention at all). Left untouched when no match is found, so
    // callers can use a sentinel like -1.0 to mean "not this reason."
    bool has_pending_hint(int layer_idx, int expert_id, double* out_queued_ms = nullptr) const {
        unsigned int h = head.load(std::memory_order_acquire);
        unsigned int t = tail.load(std::memory_order_relaxed);
        for (unsigned int i = t; i != h; ++i) {
            const PrefetchHint& hint = queue[i & (QUEUE_SIZE - 1)];
            if (hint.layer_idx == layer_idx && hint.expert_id == expert_id) {
                if (out_queued_ms) {
                    *out_queued_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - hint.pushed_at).count();
                }
                return true;
            }
        }
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

            active_prefetch(hint.layer_idx, hint.expert_id, my_ring, hint.source);
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

    // RAII: reverts an in-progress RAM-tier claim to Empty unless explicitly
    // disarmed at the point of successful publish. Without this, ANY exit
    // path after claim_or_wait returns Claimed (a geometry lookup failure, a
    // failed read) would leave the slot stuck Loading forever -- nothing
    // else can claim the same key (ram_expert_index's one-slot-per-key
    // invariant), so a future decode-thread miss for this exact expert would
    // poll past [SLOW-RAM-WAIT] and hang silently. See MoEExpertBlock::
    // ram_loading's own comment (engine.hpp) for the full invariant this
    // protects.
    struct RamClaimGuard {
        RAMSafetyManager* ram_mgr;
        int layer;
        MoEExpertBlock* slot;
        bool armed = true;
        ~RamClaimGuard() { if (armed && slot) ram_mgr->abandon_ram_claim(layer, *slot); }
    };

    // Active prefetch: claim a real RAM-tier slot via RAMSafetyManager's
    // unified claim/publish design (ram_manager.hpp's claim_or_wait), then
    // populate it directly -- no separate staging pool, no separate pinned
    // buffer, no memcpy hop. This is the core of the unification: a
    // prefetch worker's claim consults the EXACT SAME index a decode-thread
    // cache hit checks, so it is structurally impossible for this call to
    // fetch an expert that's already resident (checked-and-claimed
    // atomically under ram_mgr's own lock) -- eliminating the old
    // "Cache-hit-beat-fetch" waste category by construction, not by better
    // heuristics (see CLAUDE.md's prefetch-investigation history).
    //
    // Reuses StreamingEngine::build_expert_read_requests (engine.hpp) -- the
    // SAME helper RAMSafetyManager::get_experts_batch's own Phase B already
    // uses for its io_uring batch -- instead of the ~190-line duplicate
    // packed/per-expert read-building this function used to carry (which
    // existed only because it had to fill a SEPARATE pinned staging buffer
    // instead of writing straight into a real cache slot). This also gains
    // unpacked_expert_format support for prefetch for free, closing a gap
    // the old design had (it only ever built packed-format requests).
    //
    // CRITICAL: per claim_or_wait's own CRITICAL rule (ram_manager.hpp) --
    // this function must NEVER touch GpuTopology/cache_mutex, directly or
    // indirectly, or a cross-thread AB-BA deadlock against the decode
    // thread's own cache_mutex-then-ram_mutex nesting becomes reachable.
    // Passing RamClaimCaller::PrefetchWorker to claim_or_wait is what
    // enforces this -- do not change that argument.
    //
    // my_ring is the CALLING worker thread's own io_uring ring (see
    // worker_loop) -- never shared across threads.
    void active_prefetch(int layer_idx, int expert_id, io_uring_backend::Ring& my_ring,
                          PilotSource source = PilotSource::Other) {
        if (!engine || !ram_mgr) return;
        bool is_depth2 = (source == PilotSource::Depth2);

        RamClaimResult result = ram_mgr->claim_or_wait(*engine, layer_idx, expert_id, RamClaimCaller::PrefetchWorker);
        if (result.status != RamClaimStatus::Claimed) {
            // Already resident (Hit) or already being fetched by someone
            // else (Loading, owned by the decode thread or another prefetch
            // worker) -- structurally cannot duplicate a fetch here.
            if (auto* s = layer_stat(layer_idx)) s->claims_beaten_by_decode.fetch_add(1, std::memory_order_relaxed);
            if (is_depth2) perf().n_prefetch_claims_beaten_depth2.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        MoEExpertBlock& slot = *result.slot;
        RamClaimGuard guard{ram_mgr, layer_idx, &slot};

        auto t0 = std::chrono::high_resolution_clock::now();
        perf().n_prefetch_issued.fetch_add(1, std::memory_order_relaxed);
        if (is_depth2) perf().n_prefetch_issued_depth2.fetch_add(1, std::memory_order_relaxed);
        if (auto* s = layer_stat(layer_idx)) s->fetches_issued.fetch_add(1, std::memory_order_relaxed);
        n_fetches_in_flight.fetch_add(1, std::memory_order_relaxed);
        // Measurement only (expert_fetch_tracker.hpp) -- kept for the
        // VRAM-tier shadow-fast-path's own independent signal
        // (execution_layer.cpp); the RAM-tier race this used to detect is
        // now structurally impossible under the unified claim design (see
        // this header's own top-of-file comment on the include).
        expert_fetch_tracker().mark(layer_idx, expert_id, ExpertFetchState::InFlight);

        io_uring_backend::ReadRequest reqs[3];
        int n = 0; size_t total = 0;
        bool ok = engine->build_expert_read_requests(layer_idx, expert_id, slot, reqs, n, total);
        if (ok) {
            ok = try_uring_batch(reqs, n, my_ring);
            if (!ok) {
                bool all_ok = true;
                for (int i = 0; i < n; ++i) {
                    ssize_t r = direct_io::pread_direct_aware(
                        reqs[i].fd, reqs[i].is_direct, reqs[i].alignment, reqs[i].dst, reqs[i].len, reqs[i].offset);
                    if (r != (ssize_t)reqs[i].len) { all_ok = false; break; }
                }
                ok = all_ok;
            }
        }

        n_fetches_in_flight.fetch_sub(1, std::memory_order_relaxed);
        if (ok) {
            double dt_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            perf().n_prefetch_completed.fetch_add(1, std::memory_order_relaxed);
            perf().bytes_prefetched.fetch_add(total, std::memory_order_relaxed);
            PerfCounters::atomic_add(perf().t_prefetch_read_ms_sum, dt_ms);
            if (is_depth2) {
                perf().n_prefetch_completed_depth2.fetch_add(1, std::memory_order_relaxed);
                perf().bytes_prefetched_depth2.fetch_add(total, std::memory_order_relaxed);
                PerfCounters::atomic_add(perf().t_prefetch_read_ms_sum_depth2, dt_ms);
            }
            if (auto* s = layer_stat(layer_idx)) {
                s->fetches_completed.fetch_add(1, std::memory_order_relaxed);
                s->fetch_us_sum.fetch_add((uint64_t)(dt_ms * 1000.0), std::memory_order_relaxed);
            }

            // Set up gate_proj/up_proj/down_proj host_ptr/meta from the now-
            // populated slab_buffer -- same helper Phase B uses, works
            // identically whether 1 (unpacked) or 3 (packed) reads filled
            // the buffer, since the destination layout (gate@0, up@gb,
            // down@gb+ub) is the same either way.
            size_t gb, ub, db;
            const GGUFTensor *gM, *uM, *dM;
            if (engine->expert_layer_geometry(layer_idx, gb, ub, db, gM, uM, dM)) {
                engine->populate_packed_slab(slot, expert_id, gb, ub, db, *gM, *uM, *dM);
                slot.loaded_by_prefetch = true;
                ram_mgr->publish_ram_slot(slot);
                guard.armed = false;  // successfully published, not abandoned
                return;
            }
            // Geometry lookup failed despite a successful read -- should be
            // unreachable (build_expert_read_requests already needed the
            // same geometry to succeed), but if it ever happens, fall
            // through to the failure path below rather than publish an
            // unpopulated slot.
        }

        perf().n_prefetch_failed.fetch_add(1, std::memory_order_relaxed);
        if (is_depth2) perf().n_prefetch_failed_depth2.fetch_add(1, std::memory_order_relaxed);
        if (auto* s = layer_stat(layer_idx)) s->fetches_failed.fetch_add(1, std::memory_order_relaxed);
        issue_fadvise_fallback(layer_idx, expert_id);
        // guard stays armed -- its destructor reverts the claim to Empty.
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