// EngineRunner phases: offload_dense_tensors.
// Split from the former run_engine() in src/main.cpp; bodies are unchanged except where
// a former local became an EngineRunner member of the same name.
#include "engine_runner.hpp"

std::expected<void, std::string> EngineRunner::offload_dense_tensors() {
  gpu.report_vram_usage("after expert cache setup");

  // Connect VRAM expert cache to the RAM manager for LRU integration
  ram_mgr.vram_cache = &gpu.topology;
  // NOTE: the reverse pointer (RAMSafetyManager -> LockFreePrefetcher) no
  // longer exists -- under the unified claim/publish design the
  // prefetcher calls INTO RAMSafetyManager::claim_or_wait, not the other
  // way around, so ram_mgr itself never needs to reach the prefetcher.
  // See prefetcher.ram_mgr's own assignment further down (after the Tier-3
  // RAM preload block) for the one remaining pointer, in the opposite
  // direction.

  // Offload dense tensors to core GPU VRAM — via a single pool allocation
  // (see CudaManager::init_dense_pool/offload_tensor_to_pool) rather than
  // one cudaMalloc PER tensor. With ~1182 separate tensors of wildly
  // varying sizes, individual allocations fragment a near-full device:
  // the allocator can fail to place a tensor even when total free bytes
  // are technically sufficient, because no single gap is large enough.
  // One pool + offset suballocation has no such gaps — a tensor either
  // fits in the remaining linear space or it doesn't, with zero waste
  // between tensors beyond 256-byte alignment padding.
  LOG_STREAM_INFO()
      << "[CUDA] Offloading persistent dense layers to core GPU (device "
      << gpu.get_core_device() << ")...\n";

  // Offload attention, norms, router, shared-expert, dense leading-layer
  // FFN, and embedding/output to VRAM. These are all "resident" (always-
  // used, never streamed) tensors — unlike routed experts they should
  // never live host-only. The shared-expert weights (ffn_*_shexp — one
  // always-active FFN block PER LAYER, run on every token) and the
  // leading dense layers' plain ffn_gate/up/down (l < first_dense) were
  // previously missing from this filter entirely: they matched none of
  // the substrings below, so dense_proj's GPU dispatch check
  // (`W.device_ptr`) always failed for them and every token paid the
  // full CPU dequant-matmul cost for every layer's shared expert.
  // HC/GDN GGML runners borrow their weight pointers from this shared dense
  // pool, avoiding a second private CUDA copy. Host copies remain in
  // resident_layers for registration and CPU fallback.
  const FullGraphModel *gm = engine.family_ops.graph_model;
  const bool embedding_on_gpu = zeromoe::memory::embedding_on_gpu(
      config.embedding_device, gm && gm->embedding_on_gpu());
  LOG_STREAM_INFO() << "[CUDA] token_embd.weight stays on the " << (embedding_on_gpu ? "GPU (dense pool)" : "CPU (host copy only)")
                    << " (embedding_device=" << config.embedding_device << ")\n";
  auto is_dense_offload_candidate = [&](const std::string &name) {
    if (name == "token_embd.weight" && !embedding_on_gpu) return false;
    // attn_k_b/attn_v_b are dequantized ONCE, CPU-side, straight from
    // host_ptr into the MLA absorption cache (wuk_cache/wuv_cache,
    // attention_layer.cpp's mla_attention_absorbed_gguf) -- dense_proj
    // (and therefore device_ptr) is NEVER called for these two specific
    // tensors. A VRAM-resident copy of them would be pure waste: bytes
    // uploaded, never read from GPU by anything. Excluded explicitly
    // since they'd otherwise match the broad "attn" substring below --
    // found while investigating what the "other_attention" VRAM bucket
    // (CLAUDE.md's dense-pool-composition breakdown) actually contains.
    //
    // attn_output_a (DeepSeek-V4's grouped output projection, first
    // stage) gets EXACTLY this same treatment in dsv4_attention_gguf
    // (attention_layer.cpp): dequant_row'd ONCE into a static per-layer
    // cache (woa_cache) and consumed via raw CPU matmul() per group,
    // never through dense_proj/device_ptr -- excluded for the identical
    // reason as attn_k_b/attn_v_b. attn_output_b (the second, single
    // dense matmul stage) is NOT excluded -- it genuinely goes through
    // dense_proj, see is_attn_output below.
    if (graph_uses_native_moe_cache &&
        name.find("attn_output_a.weight") != std::string::npos)
      return true;
    if (name.find("attn_k_b.weight") != std::string::npos ||
        name.find("attn_v_b.weight") != std::string::npos ||
        name.find("attn_output_a.weight") != std::string::npos) {
      return false;
    }
    // hc_ catches DeepSeek-V4's Hyper-Connections weights: the two
    // per-layer mixing matmuls (hc_attn_fn/hc_ffn_fn -- these DO route
    // through dense_proj, hyper_connections.cpp's hc_pre_collapse) plus
    // their small bias/scale companions (hc_{attn,ffn}_{base,scale},
    // read via a raw host_ptr lookup, no GPU dispatch -- including them
    // here is harmless over-inclusion, not a correctness issue, given
    // their negligible size) and the global output_hc_{fn,base,scale}
    // (hc_collapse_output). All contain the literal substring "hc_" and
    // nothing else in this codebase's tensor vocabulary does. Previously
    // missing entirely -- hc_attn_* happened to already match the loose
    // "attn" substring below, but hc_ffn_*/output_hc_* matched NOTHING,
    // silently dropping them from the dense VRAM total on a real
    // DeepSeek-V4 model (found auditing tools/gguf_plan.cpp, which
    // mirrors this exact function, against the newly-added tensor set).
    if (name.find("hc_") != std::string::npos)
      return true;
    // indexer.proj (DeepSeek-V4.1's Lightning Indexer projection) and
    // engram_q/engram_k/engram_wkv (its Engram gate/projection weights --
    // NOT engram_embd, the giant hash-embedding table that's deliberately
    // kept mmap'd on NVMe and never reaches resident_layers at all, see
    // deepseek41_engram.cpp) are real, per-token dense_proj targets that
    // were simply missing from this allowlist. Found investigating a real
    // ~10.7s/60-token-request CPU dequant-matmul cost that expert_calls'
    // own "0 CPU" count couldn't explain -- t_cpu_dequant/t_cpu_dot are
    // shared between the MoE CPU-fallback path and dense_proj's own,
    // perf_timer.hpp's own comment on that. Individually tiny tensors, but
    // engram_wkv fires per Engram layer per token and indexer.proj per
    // CSA/HCA layer per token, so the CPU-vs-GPU gap compounds across a
    // whole decode.
    if (name.find("indexer.proj.weight") != std::string::npos ||
        name.find("engram_wkv.weight") != std::string::npos ||
        name.find("engram_q.weight") != std::string::npos ||
        name.find("engram_k.weight") != std::string::npos)
      return true;
    // ssm_/ple_ catch Qwen4Exp's GDN recurrent-layer tensors (ssm_conv1d,
    // ssm_beta/alpha, ssm_out, ssm_norm, ...) and PLE layer tensors --
    // all resident, per-token dense_proj targets, same "resident and
    // always-used" rationale as the attention set. (attn_qkv/attn_gate
    // already match "attn".)
    if (name.find("ssm_") != std::string::npos ||
        name.find("ple_") != std::string::npos)
      return true;
    return name.find("attn") != std::string::npos ||
           name.find("norm") != std::string::npos ||
           name.find("gate_inp") != std::string::npos ||
           name.find("shexp") != std::string::npos ||
           name.find("ffn_gate.weight") != std::string::npos ||
           name.find("ffn_up.weight") != std::string::npos ||
           name.find("ffn_down.weight") != std::string::npos ||
           name.find("nextn.") != std::string::npos ||
           name == "token_embd.weight" || name == "output.weight" ||
           name == "output_norm.weight";
  };

  // HC/GDN tensors are included in this same pool. Their GGML runners
  // borrow the pool's device_ptrs through an external tensor marker, so
  // excluding them would only force a second private CUDA copy. The host
  // copies remain available for registration and CPU fallback.

  // attn_output fast-path: this weight is a STATIC per-layer matrix (same
  // every token, like attn_k_b/attn_v_b) that's large enough (~69 MB in
  // Q5_K for this project's GLM-5.2 target) to hit dense_matmul_host's
  // fused per-block dequant+matmul kernel, which has no GEMV specialization
  // and is disproportionately slow at S=1 decode — measured ~623ms/token
  // for this one weight alone, ~5x its similarly-staged qproj calls (see
  // CLAUDE.md). Fix: dequantize once to BF16 and use cuBLAS (matched-type
  // BF16xBF16 GEMM, dense_matmul_bf16_matched_host) instead, same pattern
  // as the attn_k_b/attn_v_b host-side cache. Unlike those (small,
  // low-rank MLA-only matrices), attn_output is full [n_heads*v_head,
  // hidden] — even BF16 (2 bytes/elem) is ~2-3x the Q5_K original size, so
  // applying this unconditionally to every layer could still be several GB
  // across a real model's full depth. So: convert greedily, budget-checked
  // against whatever's ACTUALLY left after reserving space for every OTHER
  // dense tensor (which already works today at its original quant size and
  // must not regress) — layers that don't fit just keep using the existing
  // (slower but correct) fused Q5_K/K-quant/IQ kernel path, exactly like
  // today.
  //
  // (An F32 version of this shipped first and worked, but covered fewer
  // layers per GB of budget than BF16 does. A first BF16 attempt used
  // MISMATCHED types (BF16 weight / F32 activation) and was reverted after
  // real NaN corruption; that corruption was later root-caused to an
  // unrelated concurrency race (now fixed) and, independently, Test 9
  // (main_test.cpp) confirmed on real hardware that cuBLAS flatly rejects
  // that mismatched-type combination anyway. This is the third attempt:
  // MATCHED types (BF16 weight AND BF16 activation), verified correct in
  // isolation via Test 10 before being wired in here — see CLAUDE.md.)
  auto is_attn_output = [](const std::string &name) {
    // attn_output_b (DeepSeek-V4's grouped output projection, SECOND
    // stage -- a single genuine dense_proj matmul, unlike attn_output_a
    // above) is the direct V4 analog of attn_output.weight here: same
    // "large per-layer static matrix, same kind of dense_proj call,
    // same reason the BF16 fast-path helps" reasoning applies
    // unchanged. attn_output_a is EXCLUDED from candidacy entirely
    // (is_dense_offload_candidate above), so it never reaches this
    // check regardless.
    return name.find("attn_output.weight") != std::string::npos ||
           name.find("attn_output_b.weight") != std::string::npos;
  };

  // gate_inp (MoE router) BF16 fast-path -- converted BEFORE the regular
  // dense-pool reservation below so its savings flow through to that
  // budget, same as if it had never needed regular-pool room at all.
  // Unlike attn_output, gate_inp is ALREADY F32 in this checkpoint (not a
  // slow quantized type) -- confirmed by its measured per-tensor size
  // exactly matching hidden*n_experts*4 bytes -- so F32 activations
  // already dispatch through fast, MATCHED-type cuBLAS SGEMM today; this
  // buys VRAM (BF16 is half F32), not speed. No greedy budget math here
  // like attn_output's (gate_inp's total is small, ~0.47 GB for this
  // checkpoint) -- just try each one against real free VRAM directly.
  //
  // PRECISION NOTE, read before re-enabling if this is ever reverted:
  // gate_inp IS the MoE router's own weight -- it produces the logits
  // top-k expert selection is based on. Rounding it to BF16 doesn't just
  // add generic numerical noise the way it does for a dense OUTPUT
  // projection (e.g. attn_output); a rounded logit can change which
  // experts survive top-k at close margins -- a real, if narrow, ROUTING
  // behavior change, not just precision loss on a value nothing else
  // depends on structurally. Done here only after this tradeoff was
  // surfaced and the user explicitly asked to proceed anyway -- see
  // CLAUDE.md. If generation quality regresses in a way attn_output's
  // fast-path never did, this is the first thing to suspect and revert.
  int gate_inp_converted = 0, gate_inp_skipped = 0;
  int gate_inp_scheduler_kept = 0;
  size_t gate_inp_bf16_bytes_used = 0, gate_inp_f32_equivalent_bytes = 0;
  const bool disable_router_bf16 = [] {
    const char *value = std::getenv("ZEROMOE_DISABLE_ROUTER_BF16");
    return value && value[0] == '1';
  }();
  // llama.cpp's CPU-MoE topology keeps the router projection on CUDA and moves only
  // the routed expert planes to the CPU.  The Qwen decode scheduler consumes
  // the original device_ptr representation; it cannot use the separate
  // gpu_fast_ptr BF16 representation without changing router math/type
  // contracts.  Keep these small Qwen router tensors in the ordinary dense
  // pool when that explicitly gated scheduler is enabled.  This is deliberately
  // scoped to Qwen + CPU-affinity layers; all other models and the normal
  // Qwen path retain the existing BF16 optimization.
  auto qwen_scheduler_cpu_router = [&](const std::string &name) {
    if (!config.qwen4exp_decode_scheduler ||
        engine.family_ops.family != ModelFamily::Qwen4Exp ||
        name.rfind("blk.", 0) != 0 ||
        name.find("gate_inp.weight") == std::string::npos)
      return false;
    char *end = nullptr;
    const long layer = std::strtol(name.c_str() + 4, &end, 10);
    return end != name.c_str() + 4 && end && *end == '.' && layer >= 0 &&
           layer < engine.model_file.architecture_config.n_layers &&
           is_cpu_affinity_layer((int)layer);
  };
  for (auto &[name, tensor] : engine.resident_layers) {
    if (!tensor.host_ptr || name.find("gate_inp.weight") == std::string::npos)
      continue;
    if (qwen_scheduler_cpu_router(name)) {
      ++gate_inp_scheduler_kept;
      continue;
    }
    if (disable_router_bf16) {
      ++gate_inp_skipped;
      continue;
    }
    size_t remaining_free_vram = gpu.core_free_vram_bytes();
    if (gpu.upload_dequantized_bf16(tensor, remaining_free_vram)) {
      ++gate_inp_converted;
      gate_inp_bf16_bytes_used += (size_t)tensor.meta.shape[0] *
                                  (size_t)tensor.meta.shape[1] *
                                  sizeof(uint16_t);
      gate_inp_f32_equivalent_bytes += (size_t)tensor.meta.shape[0] *
                                       (size_t)tensor.meta.shape[1] *
                                       sizeof(float);
    } else {
      ++gate_inp_skipped;
    }
  }
  LOG_STREAM_INFO()
      << "[CUDA] gate_inp (MoE router) BF16 fast-path: " << gate_inp_converted
      << " layer(s) converted (" << gate_inp_bf16_bytes_used / 1e9 << " GB, vs "
      << gate_inp_f32_equivalent_bytes / 1e9 << " GB at original F32 -- "
      << (gate_inp_f32_equivalent_bytes - gate_inp_bf16_bytes_used) / 1e9
      << " GB saved), " << gate_inp_skipped << " layer(s) kept at original F32"
      << ", " << gate_inp_scheduler_kept
      << " Qwen scheduler router(s) deliberately kept at original F32"
      << (disable_router_bf16 ? ", router BF16 conversion disabled by env" : "")
      << " -- NOTE: rounds router logits, a routing-behavior change, not just "
         "numerical noise.\n";

  constexpr size_t POOL_ALIGN = 256;
  std::vector<QuantizedTensor *> attn_output_candidates;
  size_t non_attn_output_pool_bytes = 0;
  // Worst-case reservation: what the regular dense pool would need if NONE
  // of attn_output converts (every candidate stays at its original
  // quantized size) -- computed with the SAME aligned-accumulation as
  // non_attn_output_pool_bytes / the real pool_bytes_needed below, so this
  // reservation and the real pool sizing agree on what "fits" means.
  size_t attn_output_quantized_total = 0;
  for (auto &[name, tensor] : engine.resident_layers) {
    if (!tensor.host_ptr || !is_dense_offload_candidate(name))
      continue;
    if (tensor.gpu_fast_ptr)
      continue; // gate_inp already fast-path converted above -- needs no
                // regular-pool room
    if (is_attn_output(name)) {
      attn_output_candidates.push_back(&tensor);
      attn_output_quantized_total =
          (attn_output_quantized_total + POOL_ALIGN - 1) & ~(POOL_ALIGN - 1);
      attn_output_quantized_total += tensor.meta.size_bytes;
      continue;
    }
    non_attn_output_pool_bytes =
        (non_attn_output_pool_bytes + POOL_ALIGN - 1) & ~(POOL_ALIGN - 1);
    non_attn_output_pool_bytes += tensor.meta.size_bytes;
  }

  // BUG FIXED: this budget used to be sized against non_attn_output_pool_bytes
  // ALONE -- i.e. it implicitly assumed every attn_output layer would end up
  // converted (needing zero room in the regular pool), when in practice the
  // greedy loop below only converts as many as fit. The ones left behind
  // still need their full quantized bytes in the regular dense pool later,
  // but that need was never reserved for here -- so the regular pool
  // allocation could (and on a real run, did: "requested 13.58 GB" against
  // ~10GB actually free) come up short, spilling hundreds of tensors to the
  // CPU fallback. Fixed by reserving the WORST CASE (nothing converts) up
  // front, then spending only each conversion's NET extra cost (F32 bytes
  // minus the quantized bytes it frees from that worst-case reservation) --
  // see the loop below. This guarantees pool_bytes_needed (computed later)
  // plus every fast-path F32 allocation together can never exceed
  // free_before_fast_path - FAST_PATH_SAFETY_MARGIN, regardless of how many
  // layers convert.
  // Reserve room for a same-device dual-role CoreEngine expert-cache slice
  // (add_dual_role_expert_cache, further below) BEFORE computing the
  // attn_output BF16 budget. That slice is deliberately sized from
  // GROUND-TRUTH free VRAM measured AFTER this fast-path loop and the
  // regular dense pool have already run (see its own comment) -- which
  // means, without this reservation, BF16 (itself ground-truth-free-VRAM
  // greedy by design) claims 100% of whatever headroom exists and the
  // slice gets whatever rounding error is left over, regardless of its
  // configured moe_cache_gb value. Confirmed on a real run: 3090 as
  // dense+moe with moe_cache_gb=8.5-9.5 requested, only 0.39 GB actually
  // allocated, because BF16 converted all 78 attn_output layers (15.7 GB)
  // first. This is a target reservation, not a hard guarantee --
  // add_dual_role_expert_cache still re-checks ground-truth free VRAM
  // itself later and may end up with less if something else also grew in
  // the meantime, but BF16 can no longer be the reason it gets nothing.
  // Only reserve for the LATER add_dual_role_expert_cache path, which
  // applies solely to a CoreEngine-only device (role="dense" with a
  // supplementary moe_cache_gb slice). A genuine single-GPU role=
  // "dense+moe" device is already in vault_gpus by this point (assign_roles
  // put it there) and already had its FULL moe_cache_gb cudaMalloc'd by the
  // main allocate_expert_cache() pass before this function even started --
  // that's already subtracted out of core_free_vram_bytes() below, so
  // reserving for it AGAIN here would double-count and needlessly shrink
  // BF16's real budget.
  bool already_primary_vault_for_reservation =
      std::find(gpu.topology.vault_gpus.begin(), gpu.topology.vault_gpus.end(),
                gpu.topology.core_gpu) != gpu.topology.vault_gpus.end();
  double core_moe_cache_gb_for_reservation = 0.0;
  if (!already_primary_vault_for_reservation) {
    for (auto &gc : config.gpus) {
      if (gc.device_id == gpu.device_id) {
        core_moe_cache_gb_for_reservation = gc.moe_cache_gb;
        break;
      }
    }
  }
  size_t dual_role_slice_reservation =
      (size_t)(core_moe_cache_gb_for_reservation * 1e9);

  size_t worst_case_pool_bytes =
      non_attn_output_pool_bytes + attn_output_quantized_total;

  // Real-vs-configured dense_gb sanity check. worst_case_pool_bytes above
  // is the SAME byte-for-byte total this run's own dense pool is about to
  // request (non_attn_output_pool_bytes + attn_output at native quant
  // size) -- ground truth for THIS exact model/quant-mix, not an
  // estimate. dense_gb itself is never actually enforced as a cap
  // anywhere (confirmed by grep: only read by engine_config.hpp's
  // auto-sizing branch, which doesn't run when dense_gb is set
  // explicitly, as it is in every real config this project uses) -- so a
  // stale/wrong value doesn't break anything structurally, but it DOES
  // silently misrepresent how much of the card's VRAM budget is actually
  // spoken for, which is exactly the kind of "reasoned from a number that
  // was never true" mistake this session's own GPU1-VRAM investigation
  // kept tripping over. Matches the same fixed tools/gguf_plan.cpp
  // categorization (NextN/MTP-excluded, DSA-indexer-excluded) so a
  // pre-flight `gguf_tool plan` run and this warning agree.
  for (auto &gc : config.gpus) {
    if (gc.device_id != gpu.device_id || gc.dense_gb <= 0)
      continue;
    double real_gb = worst_case_pool_bytes / 1e9;
    double configured_gb = gc.dense_gb;
    double diff_gb = std::abs(real_gb - configured_gb);
    double diff_pct =
        (configured_gb > 0) ? 100.0 * diff_gb / configured_gb : 0.0;
    // Combined threshold, not percentage alone: on a small (16GB-class)
    // card a "modest" 5-8% miss is still ~1GB of real, unaccounted-for
    // VRAM budget -- exactly the size of gap that mattered throughout
    // this session's own GPU1 investigation. Fires on EITHER a real
    // >=5% relative miss or a >=0.5GB absolute one, so it still catches
    // a small-percentage-but-large-absolute miss on a big card too.
    if (diff_pct >= 5.0 || diff_gb >= 0.5) {
      LOG_STREAM_WARN()
          << "[CONFIG] dense_gb=" << configured_gb << " GB (device "
          << gc.device_id << ") looks stale -- this "
          << "model's REAL dense-tensor requirement (native quant, "
          << "nothing BF16-converted, gate_inp excluded since it's "
          << "budgeted/converted separately) is " << real_gb << " GB, "
          << diff_pct << "% off. This isn't a hard cap "
          << "(the pool allocates against real free VRAM regardless "
          << "of this value), but it misrepresents this card's real "
          << "budget for any headroom math you do by hand. Suggested: "
          << "dense_gb = " << (real_gb + 0.5)
          << " (real requirement + ~0.5GB margin). Cross-check with "
          << "`gguf_tool plan <model>` before a real run.\n";
    }
  }

  constexpr size_t FAST_PATH_SAFETY_MARGIN =
      (size_t)1
      << 30; // 1 GB, matches the dense-pool backoff's own margin style
  size_t free_before_fast_path = gpu.core_free_vram_bytes();
  size_t fast_path_reserved = worst_case_pool_bytes + FAST_PATH_SAFETY_MARGIN +
                              dual_role_slice_reservation;
  size_t fast_path_budget = (free_before_fast_path > fast_path_reserved)
                                ? (free_before_fast_path - fast_path_reserved)
                                : 0;

  int fast_converted = 0, fast_skipped = 0;
  int fast_skipped_budget = 0, fast_skipped_upload = 0;
  size_t fast_bytes_used =
      0; // gross BF16 bytes actually allocated (for the log line only)
  size_t budget_spent = 0; // NET budget consumed: sum of (bf16_bytes -
                           // quant_bytes) over converted layers
  // Tracks what the SAME converted layers would have cost under the old
  // F32 fast-path -- comparison-only bookkeeping, doesn't affect which
  // layers convert or how much budget is spent. Exists purely to make
  // BF16's actual saving directly visible in the log line below instead
  // of requiring hand arithmetic against a prior run's numbers (see
  // CLAUDE.md -- this was confusing enough in practice to need its own
  // explicit log line).
  size_t f32_equivalent_bytes = 0;
  for (QuantizedTensor *qt : attn_output_candidates) {
    size_t quant_bytes = qt->meta.size_bytes;
    size_t bf16_bytes_estimate = (size_t)qt->meta.shape[0] *
                                 (size_t)qt->meta.shape[1] * sizeof(uint16_t);
    size_t net_cost = (bf16_bytes_estimate > quant_bytes)
                          ? (bf16_bytes_estimate - quant_bytes)
                          : 0;
    size_t remaining_net_budget = (fast_path_budget > budget_spent)
                                      ? (fast_path_budget - budget_spent)
                                      : 0;
    if (net_cost > remaining_net_budget) {
      ++fast_skipped;
      ++fast_skipped_budget;
      continue;
    }
    // upload_dequantized_bf16 itself checks the GROSS bf16 size against
    // whatever's passed here -- pass the real remaining free VRAM (ground
    // truth), not the net-budget figure, since the two are different
    // units (net_cost already accounts for the pool-side saving that
    // ground-truth free VRAM doesn't know about).
    size_t remaining_free_vram = gpu.core_free_vram_bytes();
    if (gpu.upload_dequantized_bf16(*qt, remaining_free_vram)) {
      ++fast_converted;
      fast_bytes_used += bf16_bytes_estimate;
      budget_spent += net_cost;
      f32_equivalent_bytes +=
          (size_t)qt->meta.shape[0] * (size_t)qt->meta.shape[1] * sizeof(float);
    } else {
      ++fast_skipped;
      ++fast_skipped_upload;
    }
  }

  LOG_STREAM_INFO()
      << "[CUDA] attn_output fast-path (BF16 dequant + matched-type cuBLAS): "
      << fast_converted << " layer(s) converted (" << fast_bytes_used / 1e9
      << " GB BF16, vs " << f32_equivalent_bytes / 1e9
      << " GB if these same layers had used the old F32 fast-path -- "
      << (f32_equivalent_bytes - fast_bytes_used) / 1e9 << " GB saved), "
      << fast_skipped << " layer(s) kept on the slower fused kernel"
      << " [" << fast_skipped_budget << " skipped for lack of budget, "
      << fast_skipped_upload << " rejected by upload_dequantized_bf16 itself]"
      << " (net budget was " << fast_path_budget / 1e9
      << " GB, worst-case pool reservation " << worst_case_pool_bytes / 1e9
      << " GB).\n";
  // 2026-09-01: the two skip reasons used to be indistinguishable, which
  // is actively misleading -- the budget figure is printed either way, so a
  // run where EVERY layer was rejected by upload_dequantized_bf16 (which
  // returns false immediately for any non-F32 tensor) still reads as though
  // VRAM budget were the constraint, sending you off to tune dense_gb /
  // moe_cache_gb for nothing. On GLM-5.3 attn_output is quantized, so all
  // 78 are 'rejected', 0 are budget skips, with 15.6 GB VRAM free.
  //
  // That rejection is CORRECT, not a bug worth 'fixing': this path's
  // original win was BF16-vs-F32 (halves memory AND speeds up compute).
  // BF16-vs-quantized is the opposite trade -- 69 MB -> 201 MB per layer,
  // ~10.3 GB extra across 78 layers on a 16.65 GB card -- to accelerate a
  // ~5.7s bucket. Do not relax the F32 guard without measuring that trade.
  if (fast_skipped_upload > 0 && fast_converted == 0) {
    LOG_STREAM_INFO()
        << "[CUDA] attn_output fast-path is INACTIVE by design on this model: "
           "upload_dequantized_bf16 accepts F32 tensors only, and this model's "
           "attn_output is quantized. Not a VRAM problem -- see the note in "
           "main.cpp.\n";
  }

  // Pass 1: collect the REMAINING candidates (fast-path attn_output layers
  // excluded — they don't need a quantized VRAM copy anymore) and total the
  // pool size needed (256-byte aligned per tensor, matching
  // offload_tensor_to_pool's own alignment — must agree or the pool could
  // report "fits" here but reject tensors during the actual suballocation
  // pass below).
  // Category breakdown of what's actually going into the dense pool --
  // added after "why is this 14GB" needed real numbers instead of a
  // guess. Categorizes by tensor NAME PATTERN (most-specific check
  // first, so e.g. "attn_norm" -- which contains both "attn" and "norm"
  // -- resolves to exactly one bucket, never double-counted) and sums
  // bytes+count per category ACROSS ALL LAYERS, rather than logging one
  // line per individual layer (78 "attn_output" lines would bury the
  // signal, not reveal it) -- the category is what explains the total,
  // not which specific layer index it came from.
  struct DenseCategoryTotal {
    size_t bytes = 0;
    int count = 0;
  };
  std::map<std::string, DenseCategoryTotal> category_totals;
  auto categorize_dense_tensor = [](const std::string &name) -> const char * {
    // attn_output_a never reaches this function at all (excluded by
    // is_dense_offload_candidate above) -- only attn_output_b (V4) or
    // attn_output (V3/dense) land in this bucket.
    if (name.find("attn_output.weight") != std::string::npos ||
        name.find("attn_output_b.weight") != std::string::npos)
      return "attn_output (unconverted -- no BF16 fast-path budget left)";
    // DeepSeek-V4 Hyper-Connections weights -- see is_dense_offload_
    // candidate's own comment on why "hc_" catches all 9 (6 per-layer +
    // 3 global) of these. Checked before "attn" below since hc_attn_*
    // would otherwise fall into the misleading "other_attention"
    // bucket (they're residual-stream mixing weights, not attention
    // projections).
    if (name.find("hc_") != std::string::npos)
      return "hyper_connections (hc_*)";
    if (name.find("indexer.proj.weight") != std::string::npos ||
        name.find("engram_wkv.weight") != std::string::npos ||
        name.find("engram_q.weight") != std::string::npos ||
        name.find("engram_k.weight") != std::string::npos)
      return "indexer/engram (DeepSeek-V4.1 gate projections)";
    if (name.find("shexp") != std::string::npos)
      return "shared_expert (shexp)";
    if (name.find("ffn_gate.weight") != std::string::npos ||
        name.find("ffn_up.weight") != std::string::npos ||
        name.find("ffn_down.weight") != std::string::npos)
      return "leading_dense_ffn (pre-MoE layers)";
    if (name == "token_embd.weight" || name == "output.weight" ||
        name == "output_norm.weight")
      return "embedding + LM head";
    if (name.find("gate_inp") != std::string::npos)
      return "moe_router (gate_inp)";
    if (name.find("norm") != std::string::npos)
      return "norms";
    if (name.find("attn") != std::string::npos)
      return "other_attention (q/k/v/kv_a projections)";
    return "other";
  };

  std::vector<QuantizedTensor *> dense_candidates;
  size_t pool_bytes_needed = 0;
  for (auto &[name, tensor] : engine.resident_layers) {
    if (!tensor.host_ptr || !is_dense_offload_candidate(name))
      continue;
    if (tensor.gpu_fast_ptr)
      continue; // already fast-path converted above
    dense_candidates.push_back(&tensor);
    pool_bytes_needed =
        (pool_bytes_needed + POOL_ALIGN - 1) & ~(POOL_ALIGN - 1);
    pool_bytes_needed += tensor.meta.size_bytes;
    auto &cat = category_totals[categorize_dense_tensor(name)];
    cat.bytes += tensor.meta.size_bytes;
    cat.count += 1;
  }

  // Largest-first so the dominant contributor is obvious without having
  // to scan every line.
  std::vector<std::pair<std::string, DenseCategoryTotal>> sorted_categories(
      category_totals.begin(), category_totals.end());
  std::sort(sorted_categories.begin(), sorted_categories.end(),
            [](const auto &a, const auto &b) {
              return a.second.bytes > b.second.bytes;
            });
  LOG_STREAM_INFO() << "[VRAM-BREAKDOWN] dense pool composition (requested "
                    << pool_bytes_needed / 1e9 << " GB across "
                    << dense_candidates.size() << " tensor(s)):\n";
  for (auto &[cat_name, tot] : sorted_categories) {
    LOG_STREAM_INFO() << "  " << cat_name << ": " << tot.bytes / 1e9 << " GB ("
                      << tot.count << " tensor(s))\n";
  }

  // Largest tensors first: with a linear bump allocator, total space used
  // is the same regardless of order, but if the pool ends up smaller than
  // requested (backoff below) it's whatever's processed LAST that spills
  // to host. Prioritize keeping large, expensive-to-CPU-dequant tensors
  // (e.g. ~69 MB attn_output) on GPU over small ones (~KB norms, cheap on
  // CPU either way) by placing large tensors first in the fill order.
  std::sort(dense_candidates.begin(), dense_candidates.end(),
            [](const QuantizedTensor *a, const QuantizedTensor *b) {
              return a->meta.size_bytes > b->meta.size_bytes;
            });

  // Graphs that own all dense compute use their GGML arena. DeepSeek's
  // native graph callbacks and graph tensors share this persistent pool.
  bool pool_ready = graph_owns_dense ? true : gpu.init_dense_pool(pool_bytes_needed);
  if (graph_owns_dense) {
    LOG_STREAM_INFO() << "[CUDA] Dense pool skipped: full GGML graph owns a "
                         "GGML device arena\n";
  }
  if (!pool_ready && pool_bytes_needed > 0) {
    // Didn't fit at full size — back off to a fraction of ACTUAL free
    // VRAM (ground truth, not our own estimate) and retry. Whatever
    // doesn't fit in the reduced pool falls back to CPU dequant-matmul,
    // same graceful degradation as before, just now fragmentation-free
    // for however much DOES fit.
    size_t backoff_bytes = (size_t)(gpu.core_free_vram_bytes() * 0.95);
    LOG_STREAM_WARN() << "[CUDA] Full dense pool (" << pool_bytes_needed / 1e9
                      << " GB) didn't fit — retrying with "
                      << backoff_bytes / 1e9
                      << " GB (95% of currently free VRAM)\n";
    pool_ready = gpu.init_dense_pool(backoff_bytes);
  }

  int offload_ok = 0, offload_failed = 0;
  size_t offload_failed_bytes = 0;
  if (pool_ready && !graph_owns_dense) {
    for (QuantizedTensor *tensor : dense_candidates) {
      if (gpu.offload_tensor_to_pool(*tensor)) {
        ++offload_ok;
      } else {
        ++offload_failed;
        offload_failed_bytes += tensor->meta.size_bytes;
      }
    }
  } else if (!graph_owns_dense) {
    offload_failed = (int)dense_candidates.size();
    for (QuantizedTensor *tensor : dense_candidates)
      offload_failed_bytes += tensor->meta.size_bytes;
    LOG_STREAM_ERROR()
        << "[CUDA] Dense tensor pool could not be allocated at any size — "
        << "all " << offload_failed << " tensor(s) staying host-resident.\n";
  }

  if (graph_owns_dense) {
    LOG_STREAM_INFO() << "[CUDA] Dense offload skipped: full GGML graph owns "
                         "the device arena\n";
  } else {
    LOG_STREAM_INFO() << "[CUDA] Dense offload: " << offload_ok
                      << " tensors on VRAM\n";
  }
  if (offload_failed > 0) {
    LOG_STREAM_WARN()
        << "[CUDA] " << offload_failed << " tensor(s) ("
        << offload_failed_bytes / 1e9 << " GB) did NOT fit in VRAM — "
        << "staying host-resident (CPU dequant-matmul fallback for those). "
        << "Increase dense_gb / use a smaller-precision dense_model_path if "
           "this "
        << "matters for your workload.\n";
  }
  gpu.report_vram_usage("after dense tensor offload");
  gpu.report_vram_breakdown("after dense tensor offload", fast_bytes_used,
                            fast_converted);

  // Dual-role CoreEngine expert cache: opt-in only, via the CoreEngine
  // GPU's own [[gpus]] moe_cache_gb TOML entry (0 = disabled, the
  // default -- this NEVER activates unless explicitly configured, same
  // "0 disables" convention as pin_threshold/pin_topk_per_layer
  // elsewhere in this file). Deliberately placed HERE -- after dense
  // tensor offload AND every BF16 fast-path conversion above have
  // already claimed their real VRAM -- specifically so this sizes
  // against GROUND TRUTH remaining free VRAM (gpu.core_free_vram_bytes())
  // rather than a pre-allocation estimate: the same "trust cudaMemGetInfo
  // over any static guess" discipline this file's fast-path budgeting
  // already relies on, after this project got burned once by an estimate
  // that drifted from reality (the attn_output worst-case-reservation
  // OOM bug -- see CLAUDE.md). Must run BEFORE prefetcher.start() below:
  // add_dual_role_expert_cache's reindex is only safe before anything
  // begins actually using expert_vram_cache.
  // Skip entirely if this device is already a primary vault (role=
  // "dense+moe" on a genuine single-GPU box: assign_roles() puts it in
  // BOTH dense_gpu_idx and vault_gpus, and the main allocate_expert_cache()
  // pass already gave it its full configured moe_cache_gb, FIRST, before
  // any of the BF16/dense-offload logic above ever ran). Without this
  // guard, add_dual_role_expert_cache would attempt a SECOND, separate
  // cudaMalloc for the same configured GB on top of the pool that already
  // exists -- harmless in practice (it just fails the cap_bytes >=
  // expert_slab_bytes check below once real free VRAM is near zero and
  // logs a confusing "skipped" line right after the real pool's own
  // success line), but pointless and misleading to leave in.
  bool already_primary_vault =
      std::find(gpu.topology.vault_gpus.begin(), gpu.topology.vault_gpus.end(),
                gpu.topology.core_gpu) != gpu.topology.vault_gpus.end();
  double core_moe_cache_gb_cfg = 0.0;
  double core_moe_cache_reserve_gb = 0.0;
  for (auto &gc : config.gpus) {
    if (gc.device_id == gpu.device_id) {
      core_moe_cache_gb_cfg = gc.moe_cache_gb;
      core_moe_cache_reserve_gb = gc.moe_cache_reserve_gb;
      break;
    }
  }
  if (!already_primary_vault && expert_slab_bytes > 0 &&
      (core_moe_cache_gb_cfg > 0.0 || core_moe_cache_reserve_gb > 0.0)) {
    const bool fit = core_moe_cache_reserve_gb > 0.0;
    const size_t real_free = gpu.core_free_vram_bytes();
    const size_t reserve = fit ? size_t(core_moe_cache_reserve_gb * 1e9) : (size_t(1) << 30);
    const auto budget = zeromoe::memory::expert_cache_budget(
        real_free, real_free, size_t(core_moe_cache_gb_cfg * 1e9), reserve, fit, true);
    if (fit) {
      LOG_STREAM_INFO() << "[VRAM-FIT] device " << gpu.device_id << ": " << real_free / 1e9
                        << " GB free after dense, reserve " << core_moe_cache_reserve_gb
                        << " GB -> expert cache " << budget.bytes / 1e9
                        << " GB (moe_cache_gb=" << core_moe_cache_gb_cfg << " ignored)\n";
    }
    if (budget.bytes >= expert_slab_bytes) {
      gpu.topology.add_dual_role_expert_cache(gpu.topology.core_gpu,
                                            budget.bytes, expert_slab_bytes);
      gpu.report_vram_usage("after CoreEngine dual-role MoE cache alloc");
      gpu.report_vram_breakdown("after CoreEngine dual-role MoE cache alloc", fast_bytes_used, fast_converted);
    } else {
      LOG_STREAM_WARN() << "[GPU] CoreEngine expert cache disabled: " << real_free / 1e9
                        << " GB free, " << reserve / 1e9 << " GB reserved\n";
    }
  }

  return {};
}
