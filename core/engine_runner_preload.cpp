// EngineRunner phases: preload_experts.
// Split from the former run_engine() in src/main.cpp; bodies are unchanged except where
// a former local became an EngineRunner member of the same name.
#include "engine_runner.hpp"

// Batches up to Ring::CAPACITY/3 packed-format experts' reads into ONE
// io_uring submission, for the startup [PIN] preload pass -- real
// measurement (gguf_tool dma's own N-sweep) found that pass's original
// single-expert-at-a-time load_expert_slab loop runs at roughly the
// device's depth-1 ceiling (~2.2 GB/s) instead of its ~2.5 GB/s
// deep-queue ceiling, unlike decode-time reads this doesn't compete with
// any GPU compute for pacing, so there's nothing structurally stopping it
// from batching the same way RAMSafetyManager::get_experts_batch's own
// Phase B already does. All-or-nothing per chunk (same discipline as
// Phase B): on ANY failure (ring unavailable, batch too large, submit
// failure, short read) returns false and touches no slot's metadata --
// ensure_slab_capacity may already have grown a slab_buffer via
// build_expert_read_requests, which is harmless (grow-only, tolerated by
// a subsequent load_expert_slab call on the same slot); the caller must
// fall back to the original per-expert load_expert_slab loop for every
// item in this exact chunk.
//
// Handles BOTH packed and genuinely-unpacked (Phase 5 mixed-precision)
// layers, matching build_expert_read_requests' own layer_is_unpacked()
// branch below -- this comment previously said "packed format only,
// unpacked falls back immediately", true when build_expert_read_requests
// itself was packed-only, no longer true since it grew a real unpacked
// branch. A real bug lived here from exactly that drift: the SECOND,
// separate geometry lookup a few lines down (feeding populate_packed_slab)
// still unconditionally assumed packed, silently corrupting a mixed
// layer's smaller experts even though the real disk read right above it
// had already used the correct geometry -- see that call site's own
// comment for the full incident.
static bool preload_batch_load(
    StreamingEngine &engine, io_uring_backend::Ring &ring,
    const std::vector<std::tuple<int, int, MoEExpertBlock *>> &items) {
  if (items.empty() || !ring.available() ||
      items.size() * 3 > io_uring_backend::Ring::CAPACITY)
    return false;

  std::vector<io_uring_backend::ReadRequest> reqs;
  reqs.reserve(items.size() * 3);
  // Per-item geometry, needed to call populate_packed_slab after a
  // successful read -- build_expert_read_requests only fills the raw
  // slab_buffer bytes, not the QuantizedTensor host_ptr/meta fields (see
  // its own comment); load_expert_slab's packed-format branch calls
  // populate_packed_slab as its own last step, which this mirrors.
  struct ItemGeom {
    int layer;
    size_t gb, ub, db;
    const GGUFTensor *gM, *uM, *dM;
    int expert_id;
    MoEExpertBlock *slot;
  };
  std::vector<ItemGeom> geoms;
  geoms.reserve(items.size());

  // Stripe alternate experts across the two replicas when a mirror is
  // configured (GGUFModel::open_mirror) -- io_uring allows mixed fds in a
  // single submission, so this one submit_and_wait below is serviced by
  // BOTH devices' queues concurrently, roughly halving the wall time of
  // each chunk. Falls back to replica 0 for everything on a single-drive
  // setup (fd_for_tensor degrades internally), so no branch is needed here.
  for (auto &[layer, expert_id, slot] : items) {
    io_uring_backend::ReadRequest tmp[3];
    int cnt = 0;
    size_t total = 0;
    // Intra-expert split (see StreamingEngine::REPLICA_SPLIT): both
    // drives work on EVERY expert rather than alternating whole experts
    // between them, so even a chunk's last, otherwise-solo expert uses
    // both devices.
    int replica =
        engine.model_file.has_mirror() ? StreamingEngine::REPLICA_SPLIT : 0;
    if (!engine.build_expert_read_requests(layer, expert_id, *slot, tmp, cnt,
                                           total, replica) ||
        cnt != 3)
      return false;
    for (int j = 0; j < cnt; ++j)
      reqs.push_back(tmp[j]);
    size_t gb, ub, db;
    const GGUFTensor *gM, *uM, *dM;
    // MUST match the geometry build_expert_read_requests actually used
    // just above, not just try packed unconditionally -- a real,
    // confirmed bug (found 2026-09 chasing a real GPU-path NaN on a
    // genuinely mixed per-expert model): this function's own header
    // comment says "unpacked_expert_format/legacy per-expert-tensor
    // layouts fall back immediately (build_expert_read_requests
    // returns cnt!=3 or false for those)" -- true when this was
    // written, no longer true since build_expert_read_requests grew a
    // real layer_is_unpacked() branch (Phase 5 mixed-precision work)
    // that SUCCEEDS for an unpacked layer instead of falling back. The
    // real disk read a few lines up already correctly used the per-
    // expert unpacked geometry in that case -- but this second,
    // independent packed_slab_geometry() call still unconditionally
    // reported the layer's OLD packed baseline sizing (that tensor is
    // still physically present in a symlinked baseline shard
    // alongside the real override, same reason load_expert_slab had
    // this identical bug -- see its own comment), so a correctly-read
    // small cold expert's bytes got handed to populate_packed_slab
    // with the WRONG (larger, baseline-uniform) gate/up/down
    // boundaries, corrupting exactly where gate ends and up begins.
    if (engine.layer_is_unpacked(layer)) {
      if (!engine.unpacked_expert_geometry(layer, expert_id, gb, ub, db, gM, uM,
                                           dM))
        return false;
    } else if (!engine.packed_slab_geometry(layer, gb, ub, db, gM, uM, dM)) {
      return false;
    }
    geoms.push_back({layer, gb, ub, db, gM, uM, dM, expert_id, slot});
  }

  std::vector<ssize_t> results(reqs.size());
  // Timed and credited to the SAME t_disk_read/bytes_disk_read counters
  // load_expert_slab's own per-expert path uses -- mirrors RAMSafetyManager
  // ::get_experts_batch's Phase B convention exactly (times submit_and_wait
  // alone, charged unconditionally since a failed/short submission still
  // consumed real wall-clock time; bytes credited only on full success,
  // to avoid double-counting a chunk the caller then re-reads via the
  // load_expert_slab fallback). Without this, the shutdown report's
  // "Disk read (expert)" line would silently omit every batched chunk's
  // real disk activity -- a real gap this had before being caught by
  // comparing two real runs' own total-bytes figures.
  auto t0 = std::chrono::high_resolution_clock::now();
  bool submitted = ring.submit_and_wait(
      reqs.data(), (int)reqs.size(), results.data(), engine.drop_expert_pages);
  double batch_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0)
                        .count();
  PerfCounters::atomic_add(perf().t_disk_read, batch_ms / 1000.0);
  if (!submitted)
    return false;
  size_t total_bytes = 0;
  for (size_t r = 0; r < reqs.size(); ++r) {
    if (results[r] != (ssize_t)reqs[r].len)
      return false;
    total_bytes += reqs[r].len;
  }
  perf().bytes_disk_read.fetch_add(total_bytes, std::memory_order_relaxed);
  for (auto &g : geoms) {
    if (engine.graph_ram_layer_projection_compact(g.layer)) {
      if (!engine.populate_graph_ram_projection_views(g.layer, g.expert_id,
                                                      *g.slot))
        return false;
    } else {
      engine.populate_packed_slab(*g.slot, g.expert_id, g.gb, g.ub, g.db,
                                  *g.gM, *g.uM, *g.dM);
    }
  }
  return true;
}

  // Usage files are learned state, not model data.  A stale file from a
  // different checkpoint can therefore contain layer/expert IDs that do
  // not exist in this model.  The preload code below indexes
  // engine.layers_experts[layer] several times, so let the bad record be
  // visible and discard it before it can turn into an opaque SIGSEGV.
  // Keep this validation here, after build_model_graph has created the real
  // per-layer slot vectors (GLM-5.3 has dense leading layers and a hybrid
  // layer count), rather than trusting only architecture metadata.
std::vector<StreamingEngine::UsageEntry> EngineRunner::sanitize_usage_entries(
    const std::string &path, const char *source,
    std::vector<StreamingEngine::UsageEntry> raw) {
        const size_t model_layer_count = engine.layers_experts.size();
        const int model_expert_count =
            engine.model_file.architecture_config.n_experts;
        std::vector<StreamingEngine::UsageEntry> valid;
        valid.reserve(raw.size());
        size_t rejected = 0;
        constexpr size_t kMaxSamples = 8;
        for (size_t record = 0; record < raw.size(); ++record) {
          const auto &entry = raw[record];
          const bool layer_valid =
              entry.layer >= 0 && (size_t)entry.layer < model_layer_count;
          const size_t layer_slot_count =
              layer_valid ? engine.layers_experts[(size_t)entry.layer].size()
                          : 0;
          const bool expert_valid = entry.expert >= 0 &&
                                    entry.expert < model_expert_count &&
                                    (size_t)entry.expert < layer_slot_count;
          if (layer_valid && expert_valid) {
            valid.push_back(entry);
            continue;
          }

          ++rejected;
          if (rejected <= kMaxSamples) {
            LOG_STREAM_ERROR()
                << "[PIN-BOUNDS] Ignoring invalid " << source << " record #"
                << (record + 1) << " from " << path << ": layer=" << entry.layer
                << ", expert=" << entry.expert << ", count=" << entry.count
                << ". Model has layer vectors [0," << model_layer_count
                << ") and n_experts=" << model_expert_count
                << "; this layer has " << layer_slot_count
                << " expert slots. The usage file is stale/incompatible or "
                   "corrupt.\n";
          }
        }
        if (rejected > 0) {
          LOG_STREAM_WARN()
              << "[PIN-BOUNDS] Rejected " << rejected << "/" << raw.size()
              << " " << source << " record(s) from " << path << "; retained "
              << valid.size()
              << " model-compatible record(s). Preload will continue safely."
              << (rejected > kMaxSamples
                      ? " Additional invalid records were suppressed.\n"
                      : "\n");
        }
        return valid;
}

std::expected<void, std::string> EngineRunner::preload_experts() {
  // Graph families that own a separate expert source must not also build a
  // second cache-sized preload. Qwen/GLM use that policy unless they alias
  // the existing tiers. DeepSeek-V4.1 and opt-in Ling native graphs call the
  // cache-aware dispatcher, so they keep the ordinary usage-ranked preload
  // active in graph mode.
  if ((!graph_mode || graph_ram_cache || graph_uses_native_moe_cache) &&
      (!config.pin_file.empty() || config.ram_preload_all_experts) &&
      ram_mgr.vram_cache &&
      (ram_graph_cache || ram_mgr.vram_cache->expert_vram_cache_cap > 0)) {
    ram_mgr.vram_cache->startup_preload_phase = true;
    auto [entries, maxc] = StreamingEngine::load_usage(config.pin_file);
    const bool usage_file_had_entries = !entries.empty();
    entries =
        sanitize_usage_entries(config.pin_file, "pin_file", std::move(entries));
    maxc = 0;
    for (const auto &entry : entries)
      maxc = std::max(maxc, entry.count);
    if (entries.empty() && !config.ram_preload_all_experts) {
      if (usage_file_had_entries) {
        LOG_STREAM_WARN()
            << "[PIN] No model-compatible records remain in " << config.pin_file
            << " -- VRAM/RAM caches will fill reactively for this run.\n";
      } else {
        LOG_STREAM_INFO()
            << "[PIN] No usage file at " << config.pin_file
            << " yet (first run) -- VRAM/RAM caches will fill reactively.\n";
      }
    } else {
      std::unordered_map<int, std::vector<StreamingEngine::UsageEntry>>
          by_layer;
      for (auto &e : entries)
        by_layer[e.layer].push_back(e);
      for (auto &[layer, v] : by_layer) {
        // stable_sort: plain std::sort has no tie-break guarantee for
        // equal-count entries (very common at the low-count end of a
        // real usage file). See the prefill_pin_file merge block
        // below for the concrete, measured case this caused -- kept
        // stable here too so this ranking is deterministic on its
        // own, not just relative to that merge.
        std::stable_sort(v.begin(), v.end(), [](const auto &a, const auto &b) {
          return a.count > b.count;
        });
      }

      // prefill_pin_file: MERGE its ranking ahead of the general
      // pin_file ranking for any layer it has data for, rather than
      // replacing that layer's ranking outright. Prefill runs before
      // any decode token, so its own experts get priority for Tier
      // 1/2/3 slot assignment (they're needed first, chronologically)
      // -- but a hard per-layer REPLACE (the previous behavior) threw
      // away pin_file's own, typically much richer, decode-time
      // ranking for that same layer entirely. On a real run this
      // starved Tier 3 (RAM) down to 0 preloaded experts across every
      // layer: prefill_pin_file had only 732 known experts vs.
      // pin_file's 3311, and since decode dominates a normal
      // request's wall-clock time (prefill is a handful of tokens),
      // discarding pin_file's knowledge for every prefill-covered
      // layer left Tier 1/2 (topk + vram_lru target) with nothing
      // left over for Tier 3 to draw from, even though pin_ram_
      // fraction=1.0 was requesting the maximum.
      //
      // Fixed by concatenating instead: prefill's own ranked list
      // stays first (unchanged priority for Tier 1/2), then pin_
      // file's ranked list for that layer is appended MINUS any
      // expert prefill already covers (an unordered_set membership
      // check per layer) -- so a given (layer,expert) pair is never
      // attempted twice by two different source rankings, and Tier 3
      // gets pin_file's full remaining knowledge instead of nothing.
      // A layer prefill_pin_file has no entries for (e.g. its very
      // first run, before any prefill has been recorded into it yet)
      // is untouched and keeps using the general pin_file ranking
      // alone, exactly as before.
      int prefill_layers_merged = 0;
      // prefill_exclusive_protected: entries that got the bounded,
      // unconditional floor slot (pin_global_floor_per_layer/layer at
      // most). prefill_exclusive_competing: prefill-exclusive entries
      // that instead went into the honest count-sorted pool alongside
      // pin_file's own data -- a low number here relative to the
      // total known-experts count is expected and correct given how
      // flat prefill's own routing tends to be (see the per-layer
      // concentration diagnostic below); it's only a problem if it's
      // near-zero for EVERY layer, which would mean the floor itself
      // is mis-sized (pin_global_floor_per_layer too small/large).
      int prefill_exclusive_protected = 0, prefill_exclusive_competing = 0;
      if (!config.prefill_pin_file.empty()) {
        auto [pf_entries, pf_maxc] =
            StreamingEngine::load_usage(config.prefill_pin_file);
        (void)pf_maxc;
        const bool prefill_file_had_entries = !pf_entries.empty();
        pf_entries = sanitize_usage_entries(
            config.prefill_pin_file, "prefill_pin_file", std::move(pf_entries));
        if (pf_entries.empty()) {
          if (prefill_file_had_entries) {
            LOG_STREAM_WARN()
                << "[PIN] prefill_pin_file (" << config.prefill_pin_file
                << ") has no model-compatible records -- every layer falls "
                   "back "
                   "to the general pin_file ranking for this run.\n";
          } else {
            LOG_STREAM_INFO()
                << "[PIN] prefill_pin_file (" << config.prefill_pin_file
                << ") has no data yet -- every layer falls back to the general "
                   "pin_file "
                   "ranking for this run.\n";
          }
        } else {
          // Per-layer concentration diagnostic (the "statistics
          // for the prefill pin" this whole redesign was
          // motivated by) -- mirrors the general pin_file's own
          // [PIN-CONCENTRATION] block below, but computed on
          // prefill's OWN data before any merge with pin_file, so
          // it answers "how concentrated is PREFILL's routing,
          // independent of decode" directly rather than through
          // the combined ranking's own lens.
          {
            std::unordered_map<int, std::vector<StreamingEngine::UsageEntry>>
                pf_raw_by_layer;
            for (auto &e : pf_entries)
              pf_raw_by_layer[e.layer].push_back(e);
            double cov_sum = 0.0;
            int cov_n = 0;
            double cov_min = 2.0, cov_max = -1.0;
            int layer_min = -1, layer_max_l = -1;
            for (auto &[layer, v] : pf_raw_by_layer) {
              uint64_t total = 0, top1 = 0;
              for (auto &e : v) {
                total += e.count;
                if (e.count > top1)
                  top1 = e.count;
              }
              if (total == 0)
                continue;
              double share = (double)top1 / (double)total;
              cov_sum += share;
              ++cov_n;
              if (share < cov_min) {
                cov_min = share;
                layer_min = layer;
              }
              if (share > cov_max) {
                cov_max = share;
                layer_max_l = layer;
              }
              LOG_STREAM_DEBUG()
                  << "[PIN-PREFILL-CONCENTRATION] layer=" << layer
                  << " top-1 share=" << (share * 100.0) << "% of " << v.size()
                  << " distinct experts / " << total << " total picks\n";
            }
            if (cov_n > 0) {
              LOG_STREAM_INFO()
                  << "[PIN] prefill_pin_file own concentration: avg top-1 "
                     "share "
                  << (cov_sum / cov_n * 100.0) << "% across " << cov_n
                  << " layer(s) -- most concentrated: layer " << layer_max_l
                  << " (" << (cov_max * 100.0) << "%), flattest: layer "
                  << layer_min << " (" << (cov_min * 100.0)
                  << "%). Low values everywhere mean "
                     "prefill's own routing rarely has a clear per-layer "
                     "winner -- "
                     "expect the bounded floor below to protect little, by "
                     "design. "
                     "Full per-layer breakdown at log_level=DEBUG "
                     "([PIN-PREFILL-CONCENTRATION] lines).\n";
            }
          }
          std::unordered_map<int, std::vector<StreamingEngine::UsageEntry>>
              pf_by_layer;
          for (auto &e : pf_entries)
            pf_by_layer[e.layer].push_back(e);
          for (auto &[layer, pf_v] : pf_by_layer) {
            std::stable_sort(
                pf_v.begin(), pf_v.end(),
                [](const auto &a, const auto &b) { return a.count > b.count; });

            // Real, measured regression (2026-08-12, GLM-5.2),
            // fixed in two stages:
            //
            // Stage 1 (count corruption): the original version
            // here kept PREFILL's entry and dropped pin_file's
            // (decode's) entry for any (layer,expert) pair known
            // to both -- silently discarding decode's real,
            // much-larger-sample count in favor of prefill's
            // tiny one (a handful of hits from one short
            // prompt). Under pin_strategy=global_topn, ranking
            // is purely by that count value, so a genuinely
            // decode-hot expert that prefill also happened to
            // touch once lost its earned global rank to
            // whatever unrelated expert decode never needed --
            // measured net cost on a 51-token decode-heavy
            // generation: decode 0.68->0.57 tok/s (+11.2s wall)
            // vs. prefill 0.61->0.81 tok/s (-4.4s wall), a net
            // +6.7s loss. Fixed by never discarding pin_file's
            // own entry -- prefill only contributes entries
            // pin_file has ZERO history for.
            //
            // Stage 2 (positional bug, found while investigating
            // why prefill's OWN throughput never improved even
            // after stage 1 -- real per-layer concentration
            // check on .zeromoe_usage_prefill showed EVERY
            // layer's prefill routing is close to flat, 3.4%-
            // 11.4% top-1 share, nowhere near "an obvious
            // winner"): stage 1 still put ALL prefill-exclusive
            // entries at the front of a layer's list,
            // unconditionally, regardless of how many there
            // were or how small their own counts. The NUMBER of
            // pinned slots a layer gets (layer_pin_count below)
            // is decided by real per-entry counts, but WHICH
            // entries fill those slots was still pure array
            // position -- so on a layer with, say, 8 weak
            // prefill-exclusive picks (count 1-4 each) at the
            // front, they could still bump a genuinely strong
            // decode pick (count 50+) out of that layer's real
            // pinned budget, purely by position, not merit.
            //
            // Fixed by bounding prefill's UNCONDITIONAL
            // protection to prefill's own best
            // pin_global_floor_per_layer pick(s) (mirroring the
            // floor concept the global_topn remainder ranking
            // below already uses) -- everything else, whether
            // it came from prefill or decode, is pooled and
            // re-sorted by its own real count before being
            // sliced into tiers. A layer where prefill genuinely
            // IS concentrated naturally wins more of the
            // remainder ranking (its counts are real and
            // competitive); a flat layer correctly falls back
            // to being decided by decode's data alone. This is
            // effectively global-topk applied one level deeper,
            // to prefill's own contribution specifically, not
            // just the top-level pin_file/prefill_pin_file
            // split.
            auto it = by_layer.find(layer);
            std::unordered_set<int> decode_experts;
            if (it != by_layer.end()) {
              decode_experts.reserve(it->second.size());
              for (auto &e : it->second)
                decode_experts.insert(e.expert);
            }
            std::vector<StreamingEngine::UsageEntry>
                exclusive; // pf_v already sorted by count desc
            for (auto &e : pf_v) {
              if (decode_experts.find(e.expert) == decode_experts.end())
                exclusive.push_back(e); // prefill-exclusive only -- decode has
                                        // no data for this one
            }
            size_t protected_n =
                std::min((size_t)std::max(0, config.pin_global_floor_per_layer),
                         exclusive.size());
            std::vector<StreamingEngine::UsageEntry> merged(
                exclusive.begin(), exclusive.begin() + protected_n);
            std::vector<StreamingEngine::UsageEntry> rest(
                exclusive.begin() + protected_n, exclusive.end());
            if (it != by_layer.end())
              rest.insert(rest.end(), it->second.begin(), it->second.end());
            // stable_sort: THE fix -- this is the exact resort a
            // real run showed regressing decode from 0.68->0.62
            // tok/s. rest is `exclusive`'s remainder (often EMPTY
            // -- confirmed via a real usage-file check that this
            // prompt's prefill routing is a strict subset of
            // decode's own) plus a freshly-copied `it->second`.
            // Even when exclusive truly is empty, this is a NEW
            // vector object (different memory layout than
            // by_layer[layer]'s own in-place-sorted vector), and
            // plain std::sort gives no tie-break guarantee for
            // equal-count entries -- common at the low-count end
            // of a real usage file. Two sorts of "the same"
            // count-sorted data, on two different vector objects,
            // can legally disagree on tie order; under
            // pin_strategy=global_topn's hard top-N truncation
            // (see the [PIN] block below), that's enough to swap
            // which of several equal-count experts lands just
            // above vs. below the pin cutoff -- a real, measured
            // effect, not a theoretical one. stable_sort on an
            // already-count-sorted `it->second` preserves its
            // existing relative order for ties instead of
            // reshuffling them, so a functionally-empty merge
            // (the common case here) is now ALSO a byte-for-byte
            // no-op, matching prefill_pin_file being unset.
            std::stable_sort(
                rest.begin(), rest.end(),
                [](const auto &a, const auto &b) { return a.count > b.count; });
            merged.insert(merged.end(), rest.begin(), rest.end());
            by_layer[layer] = std::move(merged);
            prefill_exclusive_protected += (int)protected_n;
            prefill_exclusive_competing +=
                (int)(exclusive.size() - protected_n);
            ++prefill_layers_merged;
          }
          LOG_STREAM_INFO()
              << "[PIN] prefill_pin_file (" << config.prefill_pin_file << ", "
              << pf_entries.size() << " known experts): merged for "
              << prefill_layers_merged << " layer(s) -- "
              << prefill_exclusive_protected
              << " entries got the unconditional bounded floor "
                 "(pin_global_floor_per_layer="
              << config.pin_global_floor_per_layer << "/layer), "
              << prefill_exclusive_competing
              << " more prefill-exclusive entries went into honest count-based "
                 "competition "
                 "against pin_file's own data (wins depend on real "
                 "concentration, see "
                 "the [PIN] prefill_pin_file own concentration line above); "
                 "remaining "
                 "layers use the general pin_file ranking alone.\n";
        }
      }

      for (int layer = 0; layer < (int)engine.layers_experts.size(); ++layer)
        if (engine.qwen_ram_vram_only_layer(layer))
          by_layer.erase(layer);

      // A usage file is a ranking signal, not a complete model
      // manifest. When DDR is large enough to hold the whole MoE, add
      // zero-count entries for every model expert so Tier 3 preloads
      // the full set instead of leaving history-missing experts to
      // fault in during the first request. Existing counts retain
      // their ranking priority; the newly added entries simply fill
      // the remaining RAM slots.
      if (config.ram_preload_all_experts) {
        // `n_layers` is the target trunk count. When the GLM
        // NextN/MTP bridge is active, the optional final block is a
        // real routed MoE layer too and must be included here;
        // otherwise its first proposal can still fault experts from
        // SSD even when the target trunk was fully preloaded into
        // host RAM.
        const int model_layers = static_cast<int>(engine.layers_experts.size());
        const int model_experts =
            engine.model_file.architecture_config.n_experts;
        size_t added = 0;
        for (int layer = 0; layer < model_layers; ++layer) {
          if (engine.compute_expert_slab_bytes(layer) == 0 ||
              model_experts <= 0)
            continue;
          auto &v = by_layer[layer];
          std::vector<unsigned char> present((size_t)model_experts, 0);
          for (const auto &entry : v) {
            if (entry.expert >= 0 && entry.expert < model_experts)
              present[(size_t)entry.expert] = 1;
          }
          for (int expert = 0; expert < model_experts; ++expert) {
            if (present[(size_t)expert])
              continue;
            v.push_back({layer, expert, 0});
            ++added;
          }
          std::stable_sort(
              v.begin(), v.end(),
              [](const auto &a, const auto &b) { return a.count > b.count; });
        }
        LOG_STREAM_INFO()
            << "[PIN] ram_preload_all_experts: added " << added
            << " history-missing model experts to the preload manifest\n";
      }

      // Option B's static hot set is prepared from the same final ranking
      // that drives startup RAM/VRAM preload.  Keep membership in an O(1)
      // device-dispatch-side mask so cold routes can be forced to the CPU
      // without another histogram lookup in the token loop.
      //
      // In concentrated-only mode, qwen4exp_option_b_layer=-1 means:
      // inspect every layer in pin_file and commit only layers whose top-N
      // experts meet the requested historical coverage.  The baseline's
      // complete GPU block is excluded; Option B is an augmentation for the
      // CPU-side layers, not a reason to replace already-resident weights.
      engine.qwen4exp_option_b_hot_mask.clear();
      engine.qwen4exp_option_b_hot_masks.clear();
      engine.qwen4exp_option_b_layers.clear();
      engine.qwen4exp_option_b_layer_coverage.clear();
      engine.qwen4exp_option_b_committed = false;
      if (config.qwen4exp_option_b_top_n > 0) {
        const int n_experts = engine.model_file.architecture_config.n_experts;
        const int n_layers = static_cast<int>(engine.layers_experts.size());
        const int baseline_gpu_layers = std::min(
            std::max(0, qwen4exp_full_graph_device_expert_layers()), n_layers);
        const bool baseline_gpu_trailing =
            qwen4exp_full_graph_device_expert_trailing();
        auto baseline_gpu_layer = [&](int layer) {
          if (baseline_gpu_layers <= 0)
            return false;
          return baseline_gpu_trailing
                     ? layer >= n_layers - baseline_gpu_layers
                     : layer < baseline_gpu_layers;
        };

        std::vector<int> option_layers;
        if (config.qwen4exp_option_b_layer >= 0) {
          option_layers.push_back(config.qwen4exp_option_b_layer);
        } else if (!config.qwen4exp_option_b_selected_layers.empty()) {
          option_layers = config.qwen4exp_option_b_selected_layers;
        } else if (config.qwen4exp_option_b_concentrated_only) {
          option_layers.reserve(by_layer.size());
          for (const auto &[layer, entries_for_layer] : by_layer) {
            (void)entries_for_layer;
            option_layers.push_back(layer);
          }
          std::sort(option_layers.begin(), option_layers.end());
        }

        const int hot_n = std::min(config.qwen4exp_option_b_top_n,
                                   n_experts);
        if (config.qwen4exp_option_b_selected_layers.empty() &&
            config.qwen4exp_option_b_concentrated_only &&
            config.qwen4exp_option_b_max_layers > 0 && hot_n > 0) {
          auto coverage_for_layer = [&](int layer) {
            std::vector<uint64_t> counts(static_cast<size_t>(n_experts), 0);
            auto it = by_layer.find(layer);
            if (it != by_layer.end()) {
              for (const auto &entry : it->second) {
                if (entry.expert >= 0 && entry.expert < n_experts)
                  counts[static_cast<size_t>(entry.expert)] += entry.count;
              }
            }
            uint64_t total = 0;
            for (uint64_t count : counts)
              total += count;
            std::sort(counts.begin(), counts.end(), std::greater<uint64_t>());
            uint64_t covered = 0;
            for (int i = 0; i < hot_n; ++i)
              covered += counts[static_cast<size_t>(i)];
            return total == 0
                       ? 0.0
                       : static_cast<double>(covered) /
                             static_cast<double>(total);
          };
          std::stable_sort(option_layers.begin(), option_layers.end(),
                           [&](int a, int b) {
                             const double ca = coverage_for_layer(a);
                             const double cb = coverage_for_layer(b);
                             if (ca != cb)
                               return ca > cb;
                             return a < b;
                           });
          if (static_cast<int>(option_layers.size()) >
              config.qwen4exp_option_b_max_layers)
            option_layers.resize(
                static_cast<size_t>(config.qwen4exp_option_b_max_layers));
          LOG_STREAM_INFO()
              << "[OPTION-B] concentrated-only layer selection ranked top-"
              << option_layers.size() << " by top-" << hot_n
              << " traffic coverage (cap="
              << config.qwen4exp_option_b_max_layers << ")\n";
        }
        // Global pool mode: rank all selected layers' experts together and keep
        // the top K of the pool (see include/option_b_global_hot.hpp). The
        // per-layer loop below is skipped; the graph planner calls the same
        // helper, so both passes agree on every layer's hot set.
        const bool option_b_global =
            config.qwen4exp_option_b_global_top_k > 0;
        if (option_b_global) {
          std::vector<int> live_layers;
          for (int layer : option_layers) {
            if (layer < 0 || layer >= n_layers || is_cpu_affinity_layer(layer))
              continue;
            if (baseline_gpu_layer(layer)) {
              LOG_STREAM_INFO()
                  << "[OPTION-B] Layer " << layer
                  << " skipped: already covered by the complete GPU block\n";
              continue;
            }
            live_layers.push_back(layer);
          }
          // The decode planner and its VRAM estimate read pin_file alone.
          // by_layer also contains prefill-exclusive records for RAM preload;
          // including them here can change membership and disagree with the
          // compact GPU tensors' actual hot set.
          const auto sel = option_b_global::select(
              entries, live_layers, n_layers, n_experts,
              config.qwen4exp_option_b_global_top_k,
              config.qwen4exp_option_b_global_floor_per_layer,
              config.qwen4exp_option_b_top_n);
          for (const auto &[layer, ids] : sel.hot) {
            std::vector<uint8_t> hot_mask(static_cast<size_t>(n_experts), 0);
            for (int expert : ids)
              hot_mask[static_cast<size_t>(expert)] = 1;
            const double coverage = sel.layer_coverage.at(layer);
            engine.qwen4exp_option_b_hot_masks[layer] = hot_mask;
            engine.qwen4exp_option_b_layers.push_back(layer);
            engine.qwen4exp_option_b_layer_coverage[layer] = coverage;
            if (layer == config.qwen4exp_option_b_layer)
              engine.qwen4exp_option_b_hot_mask = hot_mask;
            LOG_STREAM_INFO()
                << "[OPTION-B] Layer " << layer
                << " global-pool hot set: hot=" << ids.size()
                << " cold=" << (n_experts - static_cast<int>(ids.size()))
                << " history coverage=" << (coverage * 100.0) << "%\n";
          }
          LOG_STREAM_INFO()
              << "[OPTION-B] global pool: top-"
              << config.qwen4exp_option_b_global_top_k << " of "
              << live_layers.size() << " layer(s) -> " << sel.total_hot
              << " expert(s) in " << sel.hot.size()
              << " layer(s), pooled history coverage="
              << (sel.pool_coverage * 100.0) << "% (floor="
              << config.qwen4exp_option_b_global_floor_per_layer
              << "/layer, per-layer ceiling="
              << config.qwen4exp_option_b_top_n << ")\n";
        }
        for (int layer : option_b_global ? std::vector<int>{} : option_layers) {
          if (layer < 0 || layer >= n_layers || n_experts <= 0)
            continue;
          if (baseline_gpu_layer(layer)) {
            LOG_STREAM_INFO()
                << "[OPTION-B] Layer " << layer
                << " skipped: already covered by the complete GPU block\n";
            continue;
          }

          std::vector<uint64_t> counts(static_cast<size_t>(n_experts), 0);
          auto it = by_layer.find(layer);
          if (it != by_layer.end()) {
            for (const auto &entry : it->second) {
              if (entry.expert >= 0 && entry.expert < n_experts)
                counts[static_cast<size_t>(entry.expert)] += entry.count;
            }
          }
          std::vector<int> ranked_ids(static_cast<size_t>(n_experts));
          for (int expert = 0; expert < n_experts; ++expert)
            ranked_ids[static_cast<size_t>(expert)] = expert;
          std::stable_sort(ranked_ids.begin(), ranked_ids.end(),
                           [&](int a, int b) {
                             if (counts[static_cast<size_t>(a)] !=
                                 counts[static_cast<size_t>(b)])
                               return counts[static_cast<size_t>(a)] >
                                      counts[static_cast<size_t>(b)];
                             return a < b;
                           });

          uint64_t total_count = 0, hot_count = 0;
          for (uint64_t count : counts)
            total_count += count;
          std::vector<uint8_t> hot_mask(static_cast<size_t>(n_experts), 0);
          for (int i = 0; i < hot_n; ++i) {
            const int expert = ranked_ids[static_cast<size_t>(i)];
            hot_mask[static_cast<size_t>(expert)] = 1;
            hot_count += counts[static_cast<size_t>(expert)];
          }
          const double coverage =
              total_count == 0
                  ? 0.0
                  : static_cast<double>(hot_count) /
                        static_cast<double>(total_count);
          const bool meets_coverage =
              total_count > 0 &&
              coverage >= config.qwen4exp_option_b_min_coverage;
          if (!meets_coverage) {
            LOG_STREAM_INFO()
                << "[OPTION-B] Layer " << layer
                << " remains CPU/DDR: top-" << hot_n
                << " coverage=" << (coverage * 100.0)
                << "% threshold="
                << (config.qwen4exp_option_b_min_coverage * 100.0) << "%\n";
            continue;
          }

          engine.qwen4exp_option_b_hot_masks[layer] = hot_mask;
          engine.qwen4exp_option_b_layers.push_back(layer);
          engine.qwen4exp_option_b_layer_coverage[layer] = coverage;
          if (layer == config.qwen4exp_option_b_layer)
            engine.qwen4exp_option_b_hot_mask = hot_mask;
          LOG_STREAM_INFO()
              << "[OPTION-B] Layer " << layer << " static hot set: committed"
              << " hot=" << hot_n << " cold=" << (n_experts - hot_n)
              << " history coverage=" << (coverage * 100.0)
              << "% threshold="
              << (config.qwen4exp_option_b_min_coverage * 100.0) << "%\n";
        }
        std::sort(engine.qwen4exp_option_b_layers.begin(),
                  engine.qwen4exp_option_b_layers.end());
        engine.qwen4exp_option_b_committed =
            !engine.qwen4exp_option_b_layers.empty();
        if (config.qwen4exp_option_b_concentrated_only &&
            config.qwen4exp_option_b_layer < 0) {
          LOG_STREAM_INFO()
              << "[OPTION-B] concentrated-only selection committed "
              << engine.qwen4exp_option_b_layers.size()
              << " CPU-side layer(s) from " << config.pin_file << "\n";
        }
      }

      // Persist this SAME final ranking (post prefill_pin_file
      // override) for RAMSafetyManager's runtime disk_batch_padding
      // to use as its candidate source -- see that member's own
      // comment (ram_manager.hpp) for why the persisted, cross-run
      // .zeromoe_usage signal is used here instead of live
      // StreamingEngine::route_freq. Cheap: at most a few thousand
      // ints total, copied once at startup, never touched again.
      for (auto &[layer, v] : by_layer) {
        auto &dst = ram_mgr.route_rank_by_layer[layer];
        dst.reserve(v.size());
        for (auto &e : v)
          dst.push_back(e.expert);
      }

      // Real per-layer capacity for each tier -- used both by
      // pin_topk_per_layer's "auto" mode (unchanged) and as the
      // DENOMINATOR for the fraction-based tier 2/3 knobs below, so
      // "0.1" means the same relative thing (10% of that tier's own
      // real capacity) regardless of how much VRAM/RAM a given
      // deployment actually has.
      // CPU-affinity layers have no physical VRAM region and must not
      // consume a share of the VRAM preload budget.  The compact planner
      // redistributes their bytes to GPU-executed layers, so dividing the
      // same target by all model layers leaves those newly available slots
      // empty (for example, 8 CPU layers used to cap the remaining 40
      // layers at 206 pins each even though their regions could hold ~248).
      // Keep the all-layer count for RAM policy/reporting, but use this
      // GPU-only count for every VRAM budget calculation.
      const int moe_layers = std::max(1, (int)by_layer.size());
      int vram_moe_layers = 0;
      for (const auto &[layer, v] : by_layer) {
        if (!v.empty() && !is_cpu_affinity_layer(layer))
          ++vram_moe_layers;
      }
      vram_moe_layers = std::max(1, vram_moe_layers);
      double avg_vram_slots_per_layer =
          (double)ram_mgr.vram_cache->expert_vram_cache_cap /
          (double)vram_moe_layers;

      int topk = config.pin_topk_per_layer;
      if (topk <= 0) {
        // Auto-size from the REAL slot count (expert_vram_cache_cap)
        // -- moe_cache_gb in the TOML is advisory only,
        // allocate_expert_cache always uses (total_vram - 1GB) per
        // vault GPU regardless of that budget, so sizing off the
        // config value here could over- or under-preload relative to
        // what actually fits.
        topk = std::max(1, (int)avg_vram_slots_per_layer);
      }

      // Three-tier caching: Use explicit pinned_vram_capacity instead of
      // implicit pin_topk_per_layer calculation. This allows precise control
      // over the pinned tier (targeting 1024 experts vs. current ~600).
      int max_pinned_total = config.pinned_vram_capacity;
      bool global_topn = (config.pin_strategy == "global_topn");
      if (max_pinned_total > 0 && !global_topn) {
        // Adjust topk per layer to respect the global pinned cap
        // When auto-sizing, round up the average rather than flooring it.
        // The physical compact-layer cap below trims the few layers whose
        // region is smaller, allowing the preload loop to consume the final
        // remainder slots instead of stranding one slot per layer.
        if (config.pin_topk_per_layer <= 0)
          topk = std::max(topk, (int)std::ceil(avg_vram_slots_per_layer));
        topk = std::min(topk, max_pinned_total / vram_moe_layers);
        LOG_STREAM_INFO() << "[PIN] Three-tier: capped per-layer topk to "
                          << topk << " (global cap: " << max_pinned_total
                          << " experts across " << vram_moe_layers
                          << " GPU-executed layers; " << moe_layers
                          << " total MoE layers)\n";
      }

      int vram_lru_n =
          std::max(0, (int)std::lround(config.pin_vram_lru_fraction *
                                       avg_vram_slots_per_layer));
      int ram_n = std::max(0, (int)std::lround(config.pin_ram_fraction *
                                               ram_mgr.slots_per_layer));
      bool prefix_vram = config.vram_preload_prefix_layers > 0;

      // Compact VRAM has a separate, non-interchangeable slot region
      // for each layer. A prefix request is expressed as a GLOBAL
      // budget, but planning it from that budget alone can ask for
      // every expert in a layer even when that layer's physical region
      // is much smaller. The old loop then read those excess slabs from
      // disk and emitted one "no victim" error per rejected upload.
      // Use the already-built physical slot map to cap each requested
      // layer before the preload loop starts. This applies to prefix
      // placement AND global_topn: the latter used to rank globally
      // without knowing that compact VRAM has independent per-layer
      // regions, so a hot layer could consume more than its region and
      // lose the excess as a no-victim upload rejection.
      std::unordered_map<int, int> compact_layer_capacity;
      if (ram_mgr.vram_cache->expert_compact_layout) {
        for (const auto &loc : ram_mgr.vram_cache->expert_vram_cache) {
          if (loc.pool_layer >= 0)
            ++compact_layer_capacity[loc.pool_layer];
        }
      }

      // Graph paths that own separate expert storage deliberately avoid a
      // second VRAM expert pool. GLM is the exception: its reusable graph
      // aliases the ordinary VRAM cache, so keep the usage-ranked preload
      // alive and let the graph reference those same pinned slots.
      if (graph_ram_cache && !glm_graph_aliases_vram_cache) {
        topk = 0;
        max_pinned_total = 0;
        global_topn = false;
        vram_lru_n = 0;
        prefix_vram = false;
      }

      // pin_strategy=="global_topn": replace the uniform per-layer topk
      // above with a per-layer pin count derived from a SINGLE ranking
      // across every layer's historical counts, not just this layer's
      // own. Every layer first gets a guaranteed floor
      // (pin_global_floor_per_layer, so a flat layer never drops to
      // zero protection), then whatever's left of pinned_vram_capacity
      // is spent on the globally-hottest remaining entries, wherever
      // they happen to live. Valid because every layer sees the same
      // number of routing decisions per token in this engine (one
      // top-k selection per layer per token) -- raw counts really are
      // comparable across layers, unlike e.g. differently-shaped
      // tensor byte sizes. Each layer's own list is already sorted
      // descending (see above), so a layer's globally-ranked share is
      // always a CONTIGUOUS prefix starting right after its floor --
      // a higher-count entry in the same layer can never lose the
      // global ranking to a lower-count one from another layer while
      // that lower one wins, so this never leaves a "hole" in the
      // middle of a layer's own sorted list. That's what lets the
      // existing per-layer "first N by rank" loop below stay
      // unchanged -- only N itself becomes per-layer instead of a
      // single shared topk.
      std::unordered_map<int, int> layer_pin_count;
      if (prefix_vram) {
        const int prefix_end =
            std::min(config.vram_preload_prefix_layers,
                     (int)engine.model_file.architecture_config.n_layers);
        const int effective_cap =
            max_pinned_total > 0 ? max_pinned_total
                                 : (int)(avg_vram_slots_per_layer * vram_moe_layers);
        int assigned = 0;
        for (int layer = 0; layer < prefix_end; ++layer) {
          if (is_cpu_affinity_layer(layer))
            continue;
          auto it = by_layer.find(layer);
          if (it == by_layer.end())
            continue;
          int layer_capacity = (int)it->second.size();
          if (!compact_layer_capacity.empty()) {
            auto cap_it = compact_layer_capacity.find(layer);
            layer_capacity = cap_it == compact_layer_capacity.end()
                                 ? 0
                                 : std::min(layer_capacity, cap_it->second);
          }
          const int take =
              std::min(layer_capacity, std::max(0, effective_cap - assigned));
          layer_pin_count[layer] = take;
          assigned += take;
          if (assigned >= effective_cap)
            break;
        }
        LOG_STREAM_INFO() << "[PIN] vram_preload_prefix_layers: layers [0, "
                          << prefix_end << ") assigned " << assigned << "/"
                          << effective_cap
                          << " VRAM slots in ascending layer order\n";
      } else if (global_topn) {
        int floor_n = std::max(0, config.pin_global_floor_per_layer);
        int floor_total = 0;
        for (auto &[layer, v] : by_layer) {
          if (is_cpu_affinity_layer(layer)) {
            layer_pin_count[layer] = 0;
            continue;
          }
          int f = std::min(floor_n, (int)v.size());
          if (!compact_layer_capacity.empty()) {
            auto cap_it = compact_layer_capacity.find(layer);
            f = cap_it == compact_layer_capacity.end()
                    ? 0
                    : std::min(f, cap_it->second);
          }
          layer_pin_count[layer] = f;
          floor_total += f;
        }
        struct GlobalCand {
          int layer;
          uint64_t count;
        };
        std::vector<GlobalCand> remainder;
        for (auto &[layer, v] : by_layer) {
          if (is_cpu_affinity_layer(layer))
            continue;
          int f = layer_pin_count[layer];
          for (int i = f; i < (int)v.size(); ++i)
            remainder.push_back({layer, v[i].count});
        }
        // stable_sort for the same reason as the merge block above.
        // Residual, NOT fixed here (separate, smaller-impact issue,
        // out of scope for the regression this fix targets): `by_layer`
        // is an unordered_map, so the for-loop feeding `remainder`
        // above already iterates layers in hash-bucket order, not a
        // deterministic one -- stable_sort preserves whatever that
        // input order was for count-ties, but doesn't make the input
        // order itself reproducible run-to-run. A fully deterministic
        // global ranking would need GlobalCand to also carry
        // `expert` and use (layer,expert) as an explicit tie-break
        // key, not just rely on input order -- a real, if lower-
        // priority, follow-up.
        std::stable_sort(remainder.begin(), remainder.end(),
                         [](const GlobalCand &a, const GlobalCand &b) {
                           return a.count > b.count;
                         });
        // If pinned_vram_capacity isn't set, fall back to the REAL VRAM
        // pool size (same quantity the per_layer_topk path's own "auto"
        // topk sizing uses: avg_vram_slots_per_layer * moe_layers) --
        // NOT "take everything remaining", which could try to pin every
        // expert ever routed to and blow far past physical VRAM capacity.
        int effective_cap = (max_pinned_total > 0)
                                ? max_pinned_total
                                : (int)(avg_vram_slots_per_layer * vram_moe_layers);
        int remaining_budget = std::max(0, effective_cap - floor_total);
        int take = 0;
        for (const GlobalCand &candidate : remainder) {
          if (take >= remaining_budget)
            break;
          int layer_cap = std::numeric_limits<int>::max();
          if (!compact_layer_capacity.empty()) {
            auto cap_it = compact_layer_capacity.find(candidate.layer);
            layer_cap =
                cap_it == compact_layer_capacity.end() ? 0 : cap_it->second;
          }
          int &assigned = layer_pin_count[candidate.layer];
          if (assigned >= layer_cap)
            continue;
          ++assigned;
          ++take;
        }
        LOG_STREAM_INFO() << "[PIN] global_topn: floor " << floor_n
                          << "/layer (" << floor_total << " total) + " << take
                          << " globally-ranked -> " << (floor_total + take)
                          << "/" << effective_cap
                          << (max_pinned_total > 0
                                  ? ""
                                  : " (pinned_vram_capacity unset -- capped to "
                                    "real VRAM pool size instead)")
                          << " target pins across " << vram_moe_layers
                          << " GPU-executed layer(s) (" << moe_layers
                          << " total MoE layers)\n";
      }

      // pin_core_gpu_top_n: of the experts layer_pin_count[] just
      // decided WILL be pinned, redirect the globally-hottest N of
      // them (by real count, across every layer) to land specifically
      // on the CoreEngine/dual-role GPU instead of wherever the
      // default unrestricted scan would put them -- see this field's
      // own EngineConfig comment for the full rationale. Only
      // meaningful under global_topn, since that's the only strategy
      // that already produces a coherent cross-layer ranking here
      // (per_layer_topk's own layer_max is decided inline, order-
      // dependent on the running total_pinned -- not worth
      // reconstructing a second ranking for a strategy this project
      // isn't actually running).
      int pinned = 0, vram_lru = 0, ram_preloaded = 0, attempted = 0;
      int pin_core_gpu_redirected = 0, pin_core_gpu_redirect_fallback = 0;
      int total_pinned = 0; // Three-tier: track total pinned count
      if (global_topn && config.pin_core_gpu_top_n > 0 &&
          gpu.topology.core_gpu >= 0 &&
          std::find(gpu.topology.vault_gpus.begin(),
                    gpu.topology.vault_gpus.end(),
                    gpu.topology.core_gpu) != gpu.topology.vault_gpus.end()) {
        struct FlatCand {
          int layer;
          int expert;
          uint64_t count;
        };
        std::vector<FlatCand> flat;
        for (auto &[layer, v] : by_layer) {
          int lim = std::min(layer_pin_count[layer], (int)v.size());
          for (int i = 0; i < lim; ++i)
            flat.push_back({layer, v[i].expert, v[i].count});
        }
        std::stable_sort(flat.begin(), flat.end(),
                         [](const FlatCand &a, const FlatCand &b) {
                           return a.count > b.count;
                         });
        int core_take = std::min((int)flat.size(), config.pin_core_gpu_top_n);
        LOG_STREAM_INFO()
            << "[PIN] pin_core_gpu_top_n: redirecting " << core_take
            << " of the globally-hottest already-pinned experts to land on the "
               "CoreEngine GPU (gpu="
            << gpu.topology.core_gpu
            << ")'s own dual-role "
               "slice specifically, instead of the default (unrestricted) "
               "placement -- "
               "falls back to unrestricted per-expert if that GPU's own slots "
               "fill up "
               "before all "
            << core_take << " fit\n";
        // Do this as its own DEDICATED pass, RIGHT NOW, before the
        // main by_layer loop below (Tier1/Tier2) ever runs -- not
        // just a check inside that loop. Reason, found via a real
        // run before trusting this: the main loop processes layers
        // in unordered_map hash-bucket order and does Tier2's own
        // UNRESTRICTED uploads for every layer along the way -- by
        // the time a later-processed layer's marked candidate is
        // reached, Tier2's own (unmarked) uploads for EARLIER-
        // processed layers can already have accidentally consumed
        // the CoreEngine GPU's tiny slot count via nothing more
        // than iteration-order luck (evict_lru's unrestricted scan
        // takes the first empty slot it finds, regardless of GPU).
        // Confirmed real: a first attempt with the check only
        // inside the main loop landed just 5 of 61 requested on
        // GPU1. Running this pass FIRST, before anything else has
        // touched that GPU's slots, removes the race entirely.
        // Simple per-item load (not the chunked io_uring batching
        // the main loop uses below) -- core_take is small (bounded
        // by the CoreEngine GPU's own tiny slot count), not worth
        // the extra machinery for a few dozen one-time reads.
        for (int i = 0; i < core_take; ++i) {
          // NOT counted into attempted/pinned/total_pinned here --
          // the main by_layer loop below will reach this exact
          // (layer,expert) again in its own normal position and
          // count it there (upload_expert's find_expert() returns
          // this SAME cached location, pin_expert() is safely
          // re-callable), which is the single source of truth for
          // "how many distinct experts are pinned" against
          // pinned_vram_capacity. Counting here too would double-
          // count these 61 and make the final tally read >1000
          // pinned when only 1000 distinct experts really are.
          MoEExpertBlock s;
          engine.load_expert_slab(flat[i].layer, flat[i].expert, s);
          if (!s.slab_buffer || s.slab_capacity == 0) {
            // load_expert_slab may have persistently pinned this
            // temporary VRAM-promotion slab.  Route cleanup
            // through the engine owner so CUDA unregisters the
            // range before the allocation is freed.
            engine.release_slab(s);
            continue; // counted as a normal disk-read miss when the main loop
                      // retries it
          }
          ExpertLocation *loc = gpu.topology.upload_expert(
              flat[i].layer, flat[i].expert, s.slab_buffer, s.slab_capacity,
              gpu.topology.core_gpu);
          if (loc) {
            gpu.topology.mark_startup_preloaded(flat[i].layer, flat[i].expert);
            ++pin_core_gpu_redirected;
          } else {
            ++pin_core_gpu_redirect_fallback; // that GPU's own slots are
                                              // already full
          }
          engine.release_slab(s);
        }
      }

      // Concentration diagnostic: pinning a FIXED top-K per layer only
      // helps future (unseen) prompts to the extent that layer's
      // historical routing is actually concentrated in that top-K --
      // a layer whose traffic is spread near-uniformly across most of
      // its 256 experts gains little from pinning any fixed subset,
      // since a new prompt is about as likely to route to an
      // unpinned expert as a pinned one. Computed here (not just
      // logged as a vague "top-heavy vs flat" guess) using the SAME
      // sorted-by-count vectors the preload loop below already builds
      // -- coverage = (sum of this layer's top-`topk` counts) /
      // (sum of ALL of this layer's recorded counts), i.e. "what
      // fraction of this layer's historical traffic would a HIT
      // against the pinned set actually explain". min/max identify
      // the single most concentrated and most uniform layers; the
      // full per-layer table (DEBUG level, high volume) lets you spot
      // whether flat layers cluster in a particular part of the
      // network (e.g. early vs late layers) rather than being random.
      {
        double coverage_sum = 0.0;
        int coverage_n = 0;
        double coverage_min = 2.0, coverage_max = -1.0;
        int layer_min = -1, layer_max = -1;
        for (auto &[layer, v] : by_layer) {
          uint64_t layer_total = 0;
          for (auto &e : v)
            layer_total += e.count;
          if (layer_total == 0)
            continue;
          uint64_t topk_sum = 0;
          int k = is_cpu_affinity_layer(layer)
                      ? 0
                      : ((global_topn || prefix_vram)
                             ? std::min(layer_pin_count[layer], (int)v.size())
                             : std::min(topk, (int)v.size()));
          for (int i = 0; i < k; ++i)
            topk_sum += v[i].count;
          double coverage = (double)topk_sum / (double)layer_total;
          coverage_sum += coverage;
          ++coverage_n;
          if (coverage < coverage_min) {
            coverage_min = coverage;
            layer_min = layer;
          }
          if (coverage > coverage_max) {
            coverage_max = coverage;
            layer_max = layer;
          }
          LOG_STREAM_DEBUG() << "[PIN-CONCENTRATION] layer=" << layer << " top-"
                             << k << " covers " << (coverage * 100.0) << "% of "
                             << v.size() << " distinct experts seen / "
                             << layer_total << " total historical selections\n";
        }
        if (coverage_n > 0) {
          LOG_STREAM_INFO()
              << "[PIN] "
              << (prefix_vram
                      ? "prefix-layer preload"
                      : (global_topn
                             ? "global_topn (per-layer K varies, see above)"
                             : ("Top-" + std::to_string(topk) + "/layer")))
              << " concentration: avg " << (coverage_sum / coverage_n * 100.0)
              << "% of each layer's historical "
              << "traffic covered by its pinned set -- most concentrated: "
                 "layer "
              << layer_max << " (" << (coverage_max * 100.0)
              << "%), most uniform: layer " << layer_min << " ("
              << (coverage_min * 100.0) << "%). A layer near the "
              << "low end gains little from pinning -- its routing is close to "
                 "evenly "
              << "spread, so a new prompt is about as likely to hit an "
                 "unpinned expert "
              << "as a pinned one. Full per-layer breakdown at log_level=DEBUG "
              << "([PIN-CONCENTRATION] lines).\n";
        }
      }

      // Batched io_uring preload (see preload_batch_load's own
      // comment for why): a dedicated Ring, separate from
      // RAMSafetyManager's decode_uring_ring/LockFreePrefetcher's
      // per-worker rings -- this whole [PIN] pass runs single-
      // threaded on the main thread before those are ever touched
      // (prefetcher.start() happens after this block, same as the
      // Unified RAM-tier design's own startup-ordering rule), so no
      // shared-ownership concern, just a plain local Ring.
      constexpr int PRELOAD_URING_CHUNK =
          (int)(io_uring_backend::Ring::CAPACITY / 3); // 21
      io_uring_backend::Ring preload_ring;
      // Keep a deterministic escape hatch for diagnosing model-specific
      // preload failures.  The normal path uses batched reads, while a
      // disabled batch ring exercises the original one-expert loader
      // without changing runtime inference behavior.
      const bool disable_preload_batch =
          std::getenv("ZEROMOE_DISABLE_PRELOAD_BATCH") != nullptr;
      bool preload_ring_ok = !disable_preload_batch && preload_ring.init();
      if (preload_ring_ok) {
        preload_ring.register_iowq_workers(
            config.expert_fetch_threads > 0
                ? (unsigned)config.expert_fetch_threads
                : 8u);
      }
      int preload_uring_chunks_ok = 0, preload_uring_chunks_fallback = 0;
      LOG_STREAM_INFO()
          << "[PIN] Batched io_uring preload: "
          << (preload_ring_ok
                  ? ("ring ready, " + std::to_string(PRELOAD_URING_CHUNK) +
                     " experts/submission")
                  : std::string(
                        "ring unavailable, falling back to the original "
                        "single-expert-at-a-time load for this whole pass"))
          << "\n";

      LOG_STREAM_INFO()
          << "[PIN] Preloading from " << config.pin_file << " ("
          << entries.size() << " known experts, max count " << maxc << "): "
          << (prefix_vram ? "prefix-layer ranking (see above)"
                          : (global_topn ? "global_topn ranking (see above)"
                                         : ("top-" + std::to_string(topk) +
                                            "/layer VRAM-pinned")))
          << ", next " << (prefix_vram ? 0 : vram_lru_n)
          << "/layer VRAM-unpinned, "
          << (config.ram_preload_all_experts
                  ? "all experts within each layer's RAM cap"
                  : ("next " + std::to_string(ram_n) + "/layer RAM"))
          << "...\n";
      // Previously, a preload attempt failing anywhere in this loop
      // (disk read producing no usable slab, or upload_expert
      // rejecting it) was completely silent -- a real run lost
      // ~99% of a 3300-attempt preload with zero evidence of WHERE
      // in the pipeline it failed. These distinguish "disk read
      // gave us nothing" from "GPU upload rejected a good slab"
      // (now itself logged in detail by upload_expert -- see
      // n_upload_expert_no_victim/n_upload_expert_memcpy_fail).
      int tier1_disk_fail = 0, tier1_upload_fail = 0;
      int tier2_disk_fail = 0, tier2_upload_fail = 0;
      // pin_global_tiers: per-layer VRAM-LRU and RAM preload counts from one
      // global ranking of what remains after each layer's pinned prefix.
      // The totals equal what the per-layer rule would have loaded.
      const bool global_tiers = config.pin_global_tiers && global_topn &&
                                !prefix_vram &&
                                !engine.qwen4exp_option_b_committed;
      std::unordered_map<int, int> global_lru_count, global_ram_count;
      if (global_tiers) {
        struct TierCand {
          int layer, idx;
          uint64_t count;
        };
        auto layer_ram_limit = [&](int layer) {
          if (graph_ram_cache && !engine.graph_ram_layer_allocated(layer))
            return 0;
          return std::min(ram_mgr.cap_for_layer(layer),
                          (int)engine.layers_experts[layer].size());
        };
        std::vector<int> layers;
        for (auto &[layer, v] : by_layer)
          if (layer >= 0 && (size_t)layer < engine.layers_experts.size() &&
              !engine.layers_experts[(size_t)layer].empty())
            layers.push_back(layer);
        std::sort(layers.begin(), layers.end());
        auto rank_remaining = [&](const std::unordered_map<int, int> &start,
                                  const std::unordered_map<int, int> &limit,
                                  long budget,
                                  std::unordered_map<int, int> &out) {
          std::vector<TierCand> cands;
          for (int layer : layers) {
            const auto &v = by_layer[layer];
            const int s0 = start.count(layer) ? start.at(layer) : 0;
            const int lim = limit.count(layer) ? limit.at(layer) : 0;
            for (int i = s0; i < (int)v.size() && i < s0 + lim; ++i)
              cands.push_back({layer, i, v[i].count});
          }
          std::stable_sort(cands.begin(), cands.end(),
                           [](const TierCand &a, const TierCand &b) {
                             if (a.count != b.count) return a.count > b.count;
                             if (a.layer != b.layer) return a.layer < b.layer;
                             return a.idx < b.idx;
                           });
          long taken = 0;
          for (const auto &c : cands) {
            if (taken >= budget) break;
            ++out[c.layer];
            ++taken;
          }
          return taken;
        };
        // Tier 2 (VRAM-LRU pre-warm): same total as the uniform rule.
        std::unordered_map<int, int> pin_start, vram_room;
        long lru_budget = 0;
        for (int layer : layers) {
          const int n = (int)by_layer[layer].size();
          const int pins = std::min(layer_pin_count.count(layer)
                                        ? layer_pin_count[layer] : 0, n);
          pin_start[layer] = pins;
          int room = n - pins;
          if (!compact_layer_capacity.empty()) {
            auto it = compact_layer_capacity.find(layer);
            room = std::min(room, it == compact_layer_capacity.end()
                                      ? 0 : it->second - pins);
          }
          room = std::max(0, room);
          if (is_cpu_affinity_layer(layer)) room = 0;
          vram_room[layer] = room;
          lru_budget += std::min(vram_lru_n, room);
        }
        const long lru_taken =
            rank_remaining(pin_start, vram_room, lru_budget, global_lru_count);
        // Tier 3 (RAM preload): same total as the uniform rule, starting
        // after each layer's VRAM entries.
        std::unordered_map<int, int> ram_start, ram_room;
        long ram_budget = 0;
        for (int layer : layers) {
          const int n = (int)by_layer[layer].size();
          const bool aff = is_cpu_affinity_layer(layer);
          const int s0 = aff ? 0
                             : std::min(n, pin_start[layer] +
                                               global_lru_count[layer]);
          ram_start[layer] = s0;
          const int limit = layer_ram_limit(layer);
          ram_room[layer] = std::max(0, std::min(limit, n - s0));
          ram_budget += config.ram_preload_all_experts
                            ? ram_room[layer]
                            : std::min(ram_n, ram_room[layer]);
        }
        const long ram_taken =
            rank_remaining(ram_start, ram_room, ram_budget, global_ram_count);
        int lmin = 1 << 30, lmax = 0, rmin = 1 << 30, rmax = 0;
        for (int layer : layers) {
          lmin = std::min(lmin, global_lru_count[layer]);
          lmax = std::max(lmax, global_lru_count[layer]);
          rmin = std::min(rmin, global_ram_count[layer]);
          rmax = std::max(rmax, global_ram_count[layer]);
        }
        LOG_STREAM_INFO() << "[PIN] global tiers: VRAM-LRU " << lru_taken << "/"
                          << lru_budget << " (per layer " << lmin << ".."
                          << lmax << "), RAM preload " << ram_taken << "/"
                          << ram_budget << " (per layer " << rmin << ".."
                          << rmax << "), RAM starts after VRAM entries\n";
      }
      for (auto &[layer, v] : by_layer) {
        // Defensive second barrier: all records are sanitized above,
        // but this loop is the last point before direct indexing into
        // layers_experts and should remain safe if a future manifest
        // augmentation adds entries from another source.
        if (layer < 0 || (size_t)layer >= engine.layers_experts.size() ||
            engine.layers_experts[(size_t)layer].empty()) {
          LOG_STREAM_ERROR() << "[PIN-BOUNDS] Skipping preload layer " << layer
                             << " from " << config.pin_file
                             << " immediately before cache indexing: valid "
                                "layer vectors are [0,"
                             << engine.layers_experts.size()
                             << ") and this layer has no expert slots.\n";
          continue;
        }
        int n = (int)v.size();
        // A complete prefix layer is intentionally placed in expert
        // ID order. Its membership is unchanged (all experts in
        // the layer are still loaded and pinned), but the physical
        // VRAM slots then form the regular [expert 0, expert 1, ...]
        // layout that GGML's 3-D MUL_MAT_ID tensor can view directly.
        // evict_lru takes the first available empty slot, while
        // pin_expert removes that slot from the free-list with
        // swap-and-pop. Therefore ascending IDs are required here:
        // expert 0 must occupy the first physical slot, expert 1 the
        // next one, and so on.
        // Partial prefix layers keep the historical frequency order
        // because they cannot use the direct resident-layer view.
        const bool complete_prefix_layer =
            prefix_vram && layer_pin_count[layer] >= n && n > 0;
        if (complete_prefix_layer) {
          std::stable_sort(
              v.begin(), v.end(),
              [](const auto &a, const auto &b) { return a.expert < b.expert; });
        }
        // cpu_affinity_layers (config): never pin or pre-warm VRAM
        // for these layers -- Tier 1/2 below both call
        // gpu.topology.upload_expert, so forcing layer_max=0 and
        // this layer's own vram_lru budget to 0 skips both loop
        // bodies entirely (their bounds become empty ranges) while
        // leaving Tier 3 (host RAM preload) untouched below --
        // that's the "MoE load into host DDR instead of DMA into
        // VRAM" contract for this layer. Also starts Tier 3 at rank
        // 0 (the layer's OWN hottest experts) instead of after a
        // pinned/vram_lru prefix that never happened for this layer.
        bool cpu_affinity = is_cpu_affinity_layer(layer);
        const bool option_b_active =
            engine.qwen4exp_option_b_committed &&
            config.qwen4exp_option_b_top_n > 0;
        const bool option_b_target =
            option_b_active && engine.qwen4exp_option_b_layer_enabled(layer);
        const bool option_b_graph_target =
            option_b_target && graph_mode && graph_ram_cache;
        int layer_vram_lru_n =
            (cpu_affinity || prefix_vram || option_b_target) ? 0
                                                             : vram_lru_n;
        if (global_tiers)
          layer_vram_lru_n = cpu_affinity ? 0 : global_lru_count[layer];
        // Three-tier: enforce global pinned capacity per layer.
        // global_topn already computed a final per-layer count
        // (floor + its share of the global ranking) up front; the
        // default per_layer_topk path keeps its EXACT original
        // running-total cap, unchanged.
        int layer_max;
        if (cpu_affinity || (option_b_active && !option_b_target)) {
          layer_max = 0;
        } else if (option_b_target) {
          // The full graph owns the selected hot experts in its compact
          // device arena. Do not also try to upload them through the legacy
          // cache (which is normally zero-capacity in the 22/26 topology).
          layer_max = option_b_graph_target
                          ? 0
                          : std::min(config.qwen4exp_option_b_top_n, n);
        } else if (global_topn || prefix_vram) {
          layer_max = layer_pin_count[layer];
        } else {
          layer_max = topk;
          if (max_pinned_total > 0) {
            layer_max = std::min(
                {topk, (int)v.size(), max_pinned_total - total_pinned});
          }
        }
        // Compact VRAM has a separate physical region per layer.  The
        // auto-sized top-k may round up by one, so never ask a region to
        // accept more experts than the planner actually allocated for it.
        if (!compact_layer_capacity.empty()) {
          auto cap_it = compact_layer_capacity.find(layer);
          layer_max = std::min(
              layer_max,
              cap_it == compact_layer_capacity.end() ? 0 : cap_it->second);
        }
        // Tier 2/3 start right after however many THIS layer got
        // pinned -- for per_layer_topk that's always exactly topk
        // (identical to the original code), but for global_topn it
        // varies per layer, so it can't be the shared topk anymore.
        // A cpu_affinity layer's layer_max is already 0, so
        // tier2_start correctly comes out to 0 too via the
        // global_topn branch (layer_max) regardless of topk.
        int tier2_start =
            (cpu_affinity || option_b_active || global_topn || prefix_vram)
                ? layer_max
                : topk;
        for (int i = 0; i < n && i < layer_max;) {
          int chunk_end =
              std::min(n, std::min(layer_max, i + PRELOAD_URING_CHUNK));
          int chunk_n = chunk_end - i;
          std::vector<MoEExpertBlock> scratch(chunk_n);
          bool batch_ok = false;
          if (preload_ring_ok) {
            std::vector<std::tuple<int, int, MoEExpertBlock *>> items;
            items.reserve(chunk_n);
            for (int j = 0; j < chunk_n; ++j)
              items.emplace_back(v[i + j].layer, v[i + j].expert, &scratch[j]);
            batch_ok = preload_batch_load(engine, preload_ring, items);
          }
          if (batch_ok) {
            ++preload_uring_chunks_ok;
          } else {
            ++preload_uring_chunks_fallback;
            for (int j = 0; j < chunk_n; ++j)
              engine.load_expert_slab(v[i + j].layer, v[i + j].expert,
                                      scratch[j]);
          }
          for (int j = 0; j < chunk_n; ++j) {
            ++attempted;
            MoEExpertBlock &s = scratch[j];
            if (s.slab_buffer && s.slab_capacity > 0) {
              // Plain unrestricted upload -- any expert this
              // pass reaches that pin_core_gpu_top_n already
              // placed on the CoreEngine GPU (the dedicated
              // pre-pass above) is ALREADY cached, so
              // upload_expert's own find_expert() check
              // returns that same location immediately here;
              // no need to re-check pin_to_core_gpu or pass a
              // target_gpu_id a second time (that would just
              // double-count the redirect diagnostics below
              // without changing where anything lands --
              // confirmed real on a first attempt at this).
              ExpertLocation *loc = gpu.topology.upload_expert(
                  v[i + j].layer, v[i + j].expert, s.slab_buffer,
                  s.slab_capacity, -1, true);
              if (loc &&
                  gpu.topology.pin_expert(v[i + j].layer, v[i + j].expert)) {
                gpu.topology.mark_startup_preloaded(v[i + j].layer,
                                                    v[i + j].expert);
                ++pinned;
                ++total_pinned; // Three-tier: track global pinned count
              } else if (!loc) {
                ++tier1_upload_fail;
              }
            } else {
              ++tier1_disk_fail;
            }
            engine.release_slab(s);
          }
          i = chunk_end;
        }
        // Tier 2: next vram_lru_n ranks -> VRAM, but NOT pinned --
        // a normal reactively-evictable slot, just pre-warmed
        // instead of starting cold. Reuses the exact same
        // upload_expert() call as Tier 1, just skips pin_expert().
        int tier2_end = std::min(n, tier2_start + layer_vram_lru_n);
        for (int i = tier2_start; i < tier2_end;) {
          int chunk_end = std::min(tier2_end, i + PRELOAD_URING_CHUNK);
          int chunk_n = chunk_end - i;
          std::vector<MoEExpertBlock> scratch(chunk_n);
          bool batch_ok = false;
          if (preload_ring_ok) {
            std::vector<std::tuple<int, int, MoEExpertBlock *>> items;
            items.reserve(chunk_n);
            for (int j = 0; j < chunk_n; ++j)
              items.emplace_back(v[i + j].layer, v[i + j].expert, &scratch[j]);
            batch_ok = preload_batch_load(engine, preload_ring, items);
          }
          if (batch_ok) {
            ++preload_uring_chunks_ok;
          } else {
            ++preload_uring_chunks_fallback;
            for (int j = 0; j < chunk_n; ++j)
              engine.load_expert_slab(v[i + j].layer, v[i + j].expert,
                                      scratch[j]);
          }
          for (int j = 0; j < chunk_n; ++j) {
            ++attempted;
            MoEExpertBlock &s = scratch[j];
            if (s.slab_buffer && s.slab_capacity > 0) {
              if (gpu.topology.upload_expert(v[i + j].layer, v[i + j].expert,
                                             s.slab_buffer, s.slab_capacity, -1,
                                             true)) {
                gpu.topology.mark_startup_preloaded(v[i + j].layer,
                                                    v[i + j].expert);
                ++vram_lru;
              } else {
                ++tier2_upload_fail;
              }
            } else {
              ++tier2_disk_fail;
            }
            engine.release_slab(s);
          }
          i = chunk_end;
        }
        // Tier 3: the next ranked experts -> host RAM tier directly.
        // In ram_preload_all_experts mode, use the entire effective
        // per-layer cap.  The old fixed ram_n fraction silently
        // limited this path to ~130 experts/layer even when the safe
        // RAM cache cap was materially larger, which left DDR bytes
        // unused and caused avoidable SSD misses.  Both modes remain
        // bounded by the layer's byte-aware cap and the slot-array
        // size; writes start at slot 0 while the cache is empty.
        int ram_layer_cap = std::min(ram_mgr.cap_for_layer(layer),
                                     (int)engine.layers_experts[layer].size());
        int ram_cap = config.ram_preload_all_experts
                          ? ram_layer_cap
                          : std::min(ram_n, ram_layer_cap);
        if (graph_ram_cache && !engine.graph_ram_layer_allocated(layer))
          ram_cap = 0;
        if (global_tiers)
          ram_cap = std::min(ram_layer_cap, global_ram_count[layer]);
        // 2026-08-29 (EngineConfig::ram_pin_eager_preregister):
        // ram_n itself is a FIXED fraction of slots_per_layer
        // (pin_ram_fraction), unrelated to how much of
        // ram_pinned_budget_gb is actually still available -- in
        // practice this routinely leaves real pin budget unused
        // (a real run: ~12.6GB of a 45GB budget). Extends ram_cap
        // further down this layer's OWN real ranking (v, already
        // sorted by historical count) using the SAME load path
        // just below, spending whatever budget room remains on
        // MORE real, historically-plausible experts -- not blind
        // capacity (see the reverted eager_preregister_remaining_
        // slots entry above this block for why that distinction
        // matters: an earlier generic-slot version of this idea
        // measurably backfired). Reading ram_pinned_bytes_used
        // fresh per layer means budget spent by earlier layers is
        // correctly reflected for later ones -- no separate
        // tracking needed, the atomic already does it.
        if (config.ram_pin_eager_preregister &&
            config.ram_pinned_budget_gb > 0.0 && ram_cap < ram_layer_cap) {
          size_t per_expert_bytes = engine.compute_expert_slab_bytes(layer);
          if (per_expert_bytes > 0) {
            size_t used =
                engine.ram_pinned_bytes_used.load(std::memory_order_relaxed);
            size_t budget = engine.ram_pinned_budget_bytes;
            size_t remaining = (used < budget) ? (budget - used) : 0;
            int extra = (int)(remaining / per_expert_bytes);
            if (extra > 0) {
              ram_cap = std::min(ram_cap + extra, ram_layer_cap);
            }
          }
        }
        // 2026-09-10: was `tier2_start + layer_vram_lru_n` -- RAM
        // tier's own preload started AFTER however many ranks VRAM's
        // pinned + LRU-prewarm tiers already consumed from this
        // SAME layer's ranked list, i.e. RAM and VRAM tiers were
        // treated as necessarily DISJOINT. Real, measured, TWICE-
        // confirmed incident: raising VRAM's own per-layer share
        // (pin_vram_lru_fraction=1.0, then separately
        // vram_preload_prefix_layers=15 with a raised
        // pinned_vram_capacity) both starved RAM tier for the exact
        // layers VRAM took the most from -- one run lost 9984 RAM
        // preloads (24528->14544, cache 100%->78%, real disk I/O
        // reappeared), a later one lost 7632 (24528->16896, cache
        // 100%->74%, 3.12 GB of real NVMe reads, prefill 74.18->
        // 22.53 tok/s, a 3.3x regression) purely from this.
        // Root cause: ZeroMoE's own RAM and VRAM tiers serve
        // mostly DIFFERENT consumers here (this session's own
        // investigation found qwen4exp's prefill dispatch path
        // sources almost everything from the RAM tier via
        // ram_direct/ram_staged and only checks VRAM residency via
        // a narrow per-expert D2D lookup -- see execution_layer.cpp
        // around GgmlMoeRunner::copy_expert_d2d) -- an expert being
        // VRAM-resident is not a reason to deny it a RAM-tier copy
        // too, and RAM has vastly more headroom (100 GB here vs
        // 17 GB of VRAM) to just hold both. cpu_affinity_layers
        // already special-cased exactly this (tier2_start forced
        // to 0 for those layers specifically, see this loop's own
        // comment above) -- this generalizes that same, already-
        // proven-safe pattern to every layer, unconditionally,
        // instead of only the cpu_affinity case.
        int ram_start_i = 0;
        if (global_tiers && !cpu_affinity)
          ram_start_i = std::min(n, layer_max + layer_vram_lru_n);
        int ram_count = std::min(std::max(0, n - ram_start_i), ram_cap);
        for (int off = 0; off < ram_count;) {
          int chunk_n = std::min(PRELOAD_URING_CHUNK, ram_count - off);
          bool batch_ok = false;
          if (preload_ring_ok) {
            std::vector<std::tuple<int, int, MoEExpertBlock *>> items;
            items.reserve(chunk_n);
            for (int j = 0; j < chunk_n; ++j) {
              int i = ram_start_i + off + j;
              // A graph RAM arena has one immutable slice per
              // expert ID. The normal preload path writes the
              // ranked list densely, which is correct for the
              // LRU cache but gives a shared arena the wrong
              // identity-to-slice mapping. Use the expert ID
              // as the slot whenever an arena is allocated.
              int slot_idx = engine.graph_ram_layer_allocated(layer)
                                 ? v[i].expert
                                 : off + j;
              items.emplace_back(v[i].layer, v[i].expert,
                                 &engine.layers_experts[layer][slot_idx]);
            }
            batch_ok = preload_batch_load(engine, preload_ring, items);
          }
          if (batch_ok) {
            ++preload_uring_chunks_ok;
          } else {
            ++preload_uring_chunks_fallback;
            for (int j = 0; j < chunk_n; ++j) {
              int i = ram_start_i + off + j;
              int slot_idx = engine.graph_ram_layer_allocated(layer)
                                 ? v[i].expert
                                 : off + j;
              engine.load_expert_slab(v[i].layer, v[i].expert,
                                      engine.layers_experts[layer][slot_idx]);
            }
          }
          for (int j = 0; j < chunk_n; ++j) {
            int i = ram_start_i + off + j;
            int slot_idx =
                engine.graph_ram_layer_allocated(layer) ? v[i].expert : off + j;
            ++attempted;
            MoEExpertBlock &slot = engine.layers_experts[layer][slot_idx];
            if (slot.slab_buffer && slot.slab_capacity > 0) {
              ++ram_preloaded;
              // Goes through RAMSafetyManager::preload_ram_slot
              // (takes ram_mutex internally) rather than poking
              // last_used_clock/hit_count/ram_expert_index
              // directly -- closes a real gap: this used to be
              // safe only because LockFreePrefetcher's worker
              // threads happened to not exist yet at this exact
              // point in startup (prefetcher.start() is now called
              // AFTER this whole [PIN] block finishes, specifically
              // so that timing argument becomes unnecessary rather
              // than load-bearing). preload_ram_slot also inserts
              // the ram_expert_index entry this write needs --
              // without it, every preloaded expert's first runtime
              // lookup would look like a genuine index miss
              // (indistinguishable from "really not resident"),
              // see that function's own comment.
              uint32_t ram_seed_hit_count = 0;
              if (config.ram_preload_seed_hit_count) {
                // Clamp below promote_hit_threshold so this only
                // biases RAM-tier retention, never fires a
                // reactive VRAM promotion before the slot has
                // actually been touched this session -- see
                // ram_preload_seed_hit_count's own comment.
                uint32_t cap = ram_mgr.promote_hit_threshold > 0
                                   ? ram_mgr.promote_hit_threshold - 1
                                   : 0;
                ram_seed_hit_count =
                    (uint32_t)std::min<uint64_t>(v[i].count, cap);
              }
              ram_mgr.preload_ram_slot(engine, v[i].layer, v[i].expert,
                                       (size_t)slot_idx, ram_seed_hit_count);
            }
          }
          off += chunk_n;
        }
      }
      LOG_STREAM_INFO() << "[PIN] Batched io_uring preload: "
                        << preload_uring_chunks_ok << " chunk(s) batched / "
                        << preload_uring_chunks_fallback
                        << " chunk(s) fell back to the original "
                           "single-expert-at-a-time load (ring "
                           "unavailable, a non-packed-format layer, or a "
                           "short/failed read -- see "
                           "preload_batch_load's own comment)\n";
      LOG_STREAM_INFO() << "[PIN] Preloaded " << pinned << " VRAM-pinned + "
                        << vram_lru << " VRAM-unpinned + " << ram_preloaded
                        << " RAM (of " << attempted << " attempted).\n";
      if (pin_core_gpu_redirected || pin_core_gpu_redirect_fallback) {
        LOG_STREAM_INFO() << "[PIN] pin_core_gpu_top_n: "
                          << pin_core_gpu_redirected
                          << " expert(s) landed on the CoreEngine GPU (gpu="
                          << gpu.topology.core_gpu << ") as intended, "
                          << pin_core_gpu_redirect_fallback
                          << " fell back to the default placement (that GPU's "
                             "own slots were full "
                             "by the time they were reached)\n";
      }
      if (tier1_disk_fail || tier1_upload_fail || tier2_disk_fail ||
          tier2_upload_fail) {
        LOG_STREAM_WARN()
            << "[PIN] Preload attrition -- Tier1(pinned): " << tier1_disk_fail
            << " disk-read failures, " << tier1_upload_fail
            << " GPU-upload rejections | "
            << "Tier2(unpinned): " << tier2_disk_fail << " disk-read failures, "
            << tier2_upload_fail << " GPU-upload rejections "
            << "(see [MOE-VRAM] upload_expert error lines above for the "
               "specific "
            << "cudaMemcpy/evict_lru cause of any upload rejections).\n";
      }

      // Three-tier caching: Log final pinned count
      if (max_pinned_total > 0) {
        const int physical_vram_target = std::min(
            max_pinned_total,
            (int)std::min<size_t>(
                ram_mgr.vram_cache->expert_vram_cache_cap,
                (size_t)std::numeric_limits<int>::max()));
        LOG_STREAM_INFO() << "[PIN] Three-tier: " << total_pinned
                          << " experts pinned (target: " << physical_vram_target
                          << ", configured: " << max_pinned_total
                          << ") across " << vram_moe_layers
                          << " GPU-executed layers (" << moe_layers
                          << " total MoE layers; CPU-affinity layers excluded) ("
                          << (total_pinned >= physical_vram_target
                                  ? "target met"
                                  : "below target")
                          << ")\n";
      }
    }
    ram_mgr.vram_cache->startup_preload_phase = false;

    if (ram_graph_cache) {
      int allocated = 0;
      int ready = 0;
      const int n_graph_layers = engine.model_file.architecture_config.n_layers;
      for (int layer = 0; layer < n_graph_layers; ++layer) {
        if (engine.graph_ram_layer_allocated(layer)) {
          ++allocated;
          if (engine.graph_ram_layer_ready(layer))
            ++ready;
        }
      }
      LOG_STREAM_INFO() << "[GRAPH-RAM] post-preload: " << ready << "/"
                        << allocated
                        << " arena layer(s) have the required expert-ID"
                           " mapping and host tensor views\n";

    }
  }


  // 2026-08-29: EngineConfig::ram_pin_eager_preregister's FIRST
  // implementation lived here -- RAMSafetyManager::eager_preregister_
  // remaining_slots, which eagerly grew+registered GENERIC (identity-
  // agnostic) empty slots for whatever ram_pinned_budget_gb the
  // ranking-informed [PIN] preload above left unused. Real A/B: it made
  // things WORSE, not better (active_up 0.76-0.81s -> 1.18-1.20s,
  // Active-layer source-pin coverage 62.7% -> 47.7%, "stayed pageable"
  // 808 -> 1232 slabs). Root cause: it's still greedy pre-allocation
  // against the SAME finite, shared budget the ranking-informed preload
  // and live demand-driven registration were already using efficiently
  // -- but UNINFORMED (no signal about which specific expert would ever
  // land in a given slot), so it displaced better-targeted allocations
  // with worse ones for the same fixed spend. Function kept in
  // ram_manager.hpp (unused, default off) per this project's own
  // convention for a correctly-diagnosed negative result -- see its own
  // comment and CLAUDE.md for the full A/B. Superseded by the extended,
  // ranking-informed Tier 3 preload below (same config flag, same
  // budget, but spends it on MORE real, historically-ranked experts
  // instead of blind slots).

  return {};
}
