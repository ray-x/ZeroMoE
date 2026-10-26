#pragma once

#include "mtp_speculation.hpp"
#include "mtp_speculative.hpp"

#include <random>
#include <string>
#include <vector>

class StreamingEngine;
class RAMSafetyManager;
class LockFreePrefetcher;

// Graph-native Qwen4Exp MTP generation: the draft runs as its own graph
// (MtpGraphRuntime), the target stays on the full-graph decode path, and
// every emitted token is verified by the target (models/graph/mtp_speculation.hpp).
// No layered execution is involved.
//
// Preconditions (checked, never silently relaxed): Qwen4Exp family, the
// shared MTP checkpoint loaded (mtp_enabled + mtp_model_path), the full graph
// built with qwen4exp_wide_export so prefill left the wide prompt rows in
// engine.qwen4exp_wide_prompt, and a coherent prompt (no prompt-cache reuse).
//
// `distribution` must be the distribution ordinary decoding samples from
// (penalties, temperature, top-p), so verified speculation reproduces it.
// `last_logits` are the target's logits after the last prompt token (V floats)
// and `sequence` must equal prompt_ids on entry. Returns true when this path
// owned generation to completion (including a stop requested by `emit`).
// Returns false with `error` set on any failure; the target/draft state is
// then unspecified and the request must be failed, not retried on another path.
bool qwen4exp_graph_mtp_run(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                            LockFreePrefetcher &prefetcher,
                            const std::vector<int> &prompt_ids,
                            const std::vector<float> &last_logits,
                            int max_new_tokens, int eos_id,
                            const zeromoe::graph::SpecDistribution &distribution,
                            std::vector<int> &sequence, std::mt19937 &rng,
                            const MtpEmitFn &emit, std::string &error);
