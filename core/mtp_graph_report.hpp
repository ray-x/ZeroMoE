#pragma once

// Publishes a finished speculation run: feeds the engine's MTP counters and
// writes the one-line "[MTP-GRAPH]" summary every graph-native MTP model uses.

#include "engine.hpp"
#include "logger.hpp"
#include "mtp_speculation.hpp"

inline void mtp_graph_report(const zeromoe::graph::SpecStats &stats) {
  perf().n_mtp_cycles.fetch_add(stats.cycles, std::memory_order_relaxed);
  perf().n_mtp_drafts_proposed.fetch_add(stats.drafted, std::memory_order_relaxed);
  perf().n_mtp_drafts_accepted.fetch_add(stats.accepted, std::memory_order_relaxed);
  perf().n_mtp_tokens_emitted.fetch_add(stats.emitted, std::memory_order_relaxed);
  LOG_INFO("[MTP-GRAPH] cycles=%llu drafted=%llu accepted=%llu (%.1f%%) emitted=%llu "
           "target_steps=%llu batches=%llu batch_tokens=%llu replay_tokens=%llu "
           "draft_steps=%llu restores=%llu%s\n",
           (unsigned long long)stats.cycles, (unsigned long long)stats.drafted,
           (unsigned long long)stats.accepted,
           stats.drafted ? 100.0 * double(stats.accepted) / double(stats.drafted) : 0.0,
           (unsigned long long)stats.emitted, (unsigned long long)stats.target_steps,
           (unsigned long long)stats.target_batches,
           (unsigned long long)stats.target_batch_tokens,
           (unsigned long long)stats.replay_tokens,
           (unsigned long long)stats.draft_steps, (unsigned long long)stats.restores,
           stats.truncated ? " (truncated at context capacity)" : "");
}
