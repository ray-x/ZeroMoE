// dense_dispatch.hpp — single entry point for dense weight matmuls in the
// forward path. Routes y[S,O] = x[S,I] @ W^T (W stored row-major [O,I], looked
// up by name in engine.resident_layers) to the GPU dispatcher when the weight's
// quant has a kernel, else falls back to CPU f32 matmul WITH A LOUD WARNING.
//
// The point: never silently reinterpret a quantized weight buffer as float.
// That reinterpret is the silent-garbage bug this project is eliminating — every
// dense matmul now goes through here so an unsupported quant is visible at the
// first token, not baked into every output.
#pragma once
#include "engine.hpp"
#include "cuda_manager.hpp"
#include "kernels_cpu.hpp"        // matmul
#include "kquant_dequant.hpp"     // cpu_quant_matmul (CPU fallback)
#include "logger.hpp"
#include "pilot_confirmed_set.hpp"  // publish_pilot_confirmed_resident -- execution_layer.cpp's pilot loop writes here
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <omp.h>  // omp_get_max_threads -- attn_score_threads()'s fallback

// Diagnostic kill-switch: forces every dense matmul, norm, and expert FFN onto
// the CPU reference path (libggml dequant + f32 matmul), bypassing ALL GPU
// dispatch. Enabled by EITHER the MOESTREAM_CPU_ONLY env var OR a TOML
// `cpu_only = true` (main.cpp calls enable_cpu_only_mode() after config load).
// Used to isolate whether garbage output comes from a GPU-side issue
// (async-upload race, device/stream, offload corruption, a fused kernel) vs a
// CPU math bug. If output becomes coherent with this set, the GPU path is the
// culprit; if still garbage, the bug is in the CPU math (attention/MoE/
// embedding) and is independent of where compute runs.
inline bool& cpu_only_flag() {
    static bool v = (std::getenv("MOESTREAM_CPU_ONLY") != nullptr);
    return v;
}
inline bool cpu_only_mode() { return cpu_only_flag(); }
inline void enable_cpu_only_mode() { cpu_only_flag() = true; }

// --- moe_prefill_batched (EngineConfig, default OFF) ---------------------
// Groups a prefill batch's tokens BY EXPERT before dispatch, so each routed
// expert runs ONE call over all the tokens routed to it, instead of one
// S=1 call per (token, expert). Same function-local-static-storage idiom as
// cpu_only_flag() above: set once at startup from config, read on a hot
// path, never mutated afterwards.
//
// WHY (measured 2026-09-04, qwen4exp): the per-token loop issues exactly
// S*n_layers*n_expert_used dispatches -- 1,530,720 for a 3,189-token
// prompt -- and each one re-reads that expert's whole ~3 MB weight slab.
// Prefill throughput is consequently FLAT in prompt length (5.4 tok/s at
// S=24, 6.02 tok/s at S=3189). Grouping makes each expert's weights be read
// once per layer instead of once per token; at S=3189/512 experts/top-10
// that is ~62 tokens per expert, and the measured gap vs llama.cpp (which
// does exactly this grouping, via ggml_mul_mat_id) is ~60x. See CLAUDE.md.
//
// !! SHARED PATH -- NOT qwen4exp-specific !! moe_forward_gguf has no family
// branching at all, so this flag's code path is reachable by GLM-5.3 and
// DeepSeek-V4 too. DEFAULT ON as of 2026-09-04, after verifying GLM-5.3 as
// well as qwen4exp. Set moe_prefill_batched=false to fall back to the
// byte-for-byte-unchanged per-token loop. When validating a NEW model, the
// check to use is a cpu_affinity_layers=[0..n] byte-identical token-stream
// A/B -- that is what was used for qwen4exp.
inline bool& moe_prefill_batched_flag() {
    static bool v = false;
    return v;
}
// GPU-side routed-expert accumulation -- see cuda/expert_accum.cu.
inline bool &moe_gpu_accum_flag() { static bool f = false; return f; }
inline void set_moe_gpu_accum(bool v) { moe_gpu_accum_flag() = v; }
inline bool moe_gpu_accum() { return moe_gpu_accum_flag(); }

// expert_gpu_activation: compute the routed-expert SwiGLU on the DEVICE
// instead of downloading gate/up, running silu/mul/clamp on the CPU, and
// uploading the product back. That round trip measured dn=13.68s of a 48s
// 512-token prefill against cmp=0.20s of actual expert matmul.
// See cuda/expert_activation.cu.
inline bool &expert_gpu_activation_flag() { static bool f = false; return f; }
inline void set_expert_gpu_activation(bool v) { expert_gpu_activation_flag() = v; }
inline bool expert_gpu_activation() { return expert_gpu_activation_flag(); }

inline bool moe_prefill_batched() { return moe_prefill_batched_flag(); }
inline void set_moe_prefill_batched(bool v) { moe_prefill_batched_flag() = v; }

// --- moe_decode_batched (EngineConfig, default OFF) -----------------------
// Decode-time (S==1) counterpart to moe_prefill_batched, from the
// MoEStream-vs-llama.cpp decode-gap investigation (2026-09-10). A live
// measurement found the `expert` dispatch bucket at ~77% of decode wall
// time, with the actual GPU matmul inside it only ~7% of that bucket --
// the rest is dispatch_expert_k's per-expert VRAM-cache bookkeeping
// (find_expert's index lookup, expert_ffn_resolve_gpu_loc's
// hold_for_compute lock, the gpu_pool issue/finish machinery), paid fresh
// for EVERY one of topk x n_layers expert lookups, every decode token --
// even when the whole expert set is already 100% RAM-resident
// (ram_preload_all_experts) and no miss is possible for the session's
// lifetime. llama.cpp instead decide weight placement ONCE at
// load time (`-ncmoe`) and never revisit it per token (see this
// project's own [[llamacpp-op-offload-is-the-lever]] /
// [[moe-expert-dispatch-not-batched]] memory).
//
// When on, dispatch_expert_k (execution_layer.cpp) skips straight to its
// OWN existing CPU-fallback branch (ram_pin -> expert_ffn_cpu ->
// ram_unpin -> accumulate, the exact same call already used whenever GPU
// dispatch fails today) for S==1, instead of attempting the VRAM path
// first. The RAM-tier pin/claim discipline (get_experts_batch, ram_pin)
// is NOT touched or weakened -- that machinery protects against genuinely
// concurrent mutation from prefetch-worker threads, not just this
// function's own redundant work, and stays exactly as today.
//
// REAL TRADE-OFF, not a free win: this also means a decode-time expert
// dispatched this way never goes through expert_ffn_resolve_gpu_loc's own
// cold-miss VRAM promotion path, so it can no longer newly promote an
// expert into VRAM residency from a decode-time hit while this is
// enabled. Acceptable when the model is already fully RAM-resident
// up front (this project's current tuned qwen4exp configs); a real
// regression risk for a config that relies on decode-time promotion to
// build VRAM residency over a session. Never engages for a vram_shadow
// block (VRAM-only, no host bytes without a synchronous reload) --
// dispatch_expert_k falls through to its existing path unchanged there.
//
// VERIFIED ON REAL HARDWARE, 2026-09-10, qwen4exp (Qwen3.8-Flash-Next,
// UD-IQ1_M) -- MUST STAY FALSE FOR THIS MODEL. Position-controlled A/B
// (baseline / batched / baseline, same build, same 1-token prompt):
//   baseline (GPU dispatch)      5.03, 4.97 tok/s decode  (~1% spread)
//   moe_decode_batched=true      2.93 tok/s decode         (-41%)
// A real, reproducible REGRESSION, not a wash and not a win.
// `expert_calls` flips from 480GPU/0CPU (baseline) to 418GPU/2462CPU
// (batched): this model's real quant type (IQ1_M) has a fast, working GPU
// kernel, so forcing CPU dequant-then-matmul (cpu_quant_matmul) instead
// is strictly worse -- it reconfirms an earlier, under-weighted project
// measurement (CPU-only decode 0.80 tok/s vs GPU-routed 1.55 tok/s,
// qwen4exp-gpu-path-state.md) that this same investigation had initially
// discounted. See memory: moe-decode-batched-regression.md for the full
// writeup, including the root-cause lesson (the `dispatch_k`/
// `phase3_loop` sub-timers this flag was built to eliminate are
// explicitly commented "upper-bound, overlaps gpu/cpu" -- they measure a
// window that OVERLAPS async GPU dispatch, not pure serial overhead
// sitting on top of it, so removing the GPU path traded a cheap async GPU
// op for an expensive synchronous CPU one, not free time back).
//
// This flag is only a plausible win when GPU dispatch was NEVER going to
// be used anyway -- i.e. the routed expert's quant type has NO GPU
// kernel at all (CLAUDE.md's CPU AVX-512 fallback list: IQ2_XXS,
// IQ3_XXS, IQ2_S) or the layer is `cpu_affinity_layers`-pinned. Qwen4Exp
// UD-IQ1_M is neither -- every one of its routed experts already has a
// fast GPU kernel (confirmed by the 480GPU/0CPU baseline above), so this
// flag has NO valid use case for that model/quant combination. Left in
// the tree, default false, as a documented negative result -- do not
// re-enable for qwen4exp without re-measuring, and do not assume this
// generalizes to a model/quant where the GPU kernel is missing or slow.
inline bool &moe_decode_batched_flag() { static bool v = false; return v; }
inline bool moe_decode_batched() { return moe_decode_batched_flag(); }
inline void set_moe_decode_batched(bool v) { moe_decode_batched_flag() = v; }

// --- moe_prefill_stream (EngineConfig, default OFF) ----------------------
// Optional legacy path that streams prefill expert weights straight from a
// whole-shard mmap into a pinned, double-buffered chunk on the GPU, bypassing
// the RAM tier and the VRAM LRU entirely. The GGML MUL_MAT_ID path does not
// require this flag: it prefers resident RAM slabs and only falls back to
// mmap when this flag explicitly enabled the mappings.
// See include/prefill_expert_stream.hpp for the measurements that motivate
// it -- in short, during prefill every expert is touched exactly once, so
// the cache reports 0% hit by arithmetic and all of its machinery is
// overhead. Default OFF and verified only on qwen4exp: moe_forward_gguf is
// family-agnostic, so GLM-5.3/DeepSeek-V4 reach this code too.
inline bool& moe_prefill_stream_flag() {
  static bool v = false;
  return v;
}
inline bool moe_prefill_stream() { return moe_prefill_stream_flag(); }
inline void set_moe_prefill_stream(bool v) { moe_prefill_stream_flag() = v; }
inline int& moe_prefill_stream_chunk_flag() {
  static int v = 64;
  return v;
}
inline int moe_prefill_stream_chunk() { return moe_prefill_stream_chunk_flag(); }

// --- moe_ggml_mul_mat_id (EngineConfig, default OFF) ---------------------
// Runs the whole routed-expert FFN for a prefill layer as ONE ggml graph of
// three GGML_OP_MUL_MAT_ID nodes (see include/ggml_moe_backend.hpp), instead
// of Phase 3b's own per-expert-bucket loop.
//
// Phase 3b already groups tokens by expert -- that is the same restructuring
// mul_mat_id performs -- but it still returns to the host once PER EXPERT for
// a residency decision, a stage/H2D, an activation gather, a dispatch and a
// sync. ggml-cuda does the row sorting and the expert bounds on the GPU and
// stays inside one graph, so this flag is the A/B for how much that remaining
// host orchestration actually costs.
//
// Requires moe_prefill_batched, a build with MOESTREAM_USE_GGML_CUDA, and a
// layer whose gate/up/down quant
// types are uniform across its experts. Any of those failing falls back to
// the existing path for that layer, never to a wrong result.
inline bool& moe_ggml_mul_mat_id_flag() {
  static bool v = false;
  return v;
}
inline bool moe_ggml_mul_mat_id() {
  static const bool env = [] {
    const char *e = std::getenv("MOESTREAM_GGML_MUL_MAT_ID");
    return e && e[0] == '1';
  }();
  return env || moe_ggml_mul_mat_id_flag();
}
inline void set_moe_ggml_mul_mat_id(bool v) { moe_ggml_mul_mat_id_flag() = v; }

// --- moe_ggml_shared_expert (Qwen4Exp prefill, default OFF) -------------
// Adds Qwen4Exp's always-on shared expert to the routed result on the same
// device graph before HC post-combine.  It removes the three dense projection
// host boundaries, CPU SwiGLU, and host residual loop from each batched MoE
// layer.  Requires the routed GGML path and a CUDA build; unsupported layouts
// fall back to the existing host shared-expert code.
inline bool &moe_ggml_shared_expert_flag() { static bool v = false; return v; }
inline bool moe_ggml_shared_expert() {
  static const bool env = [] {
    const char *e = std::getenv("MOESTREAM_GGML_SHARED_EXPERT");
    return e && e[0] == '1';
  }();
  return env || moe_ggml_shared_expert_flag();
}
inline void set_moe_ggml_shared_expert(bool v) {
  moe_ggml_shared_expert_flag() = v;
}

// --- moe_ggml_gpu_router (Qwen4Exp prefill, default OFF) -----------------
// Runs Qwen4Exp's softmax router and top-k selection on the same GGML CUDA
// backend as the HC/MoE graphs. The HC pre-mix result remains on the device
// while routing; only [topk, tokens] IDs and weights return to C++ for the
// existing expert-cache bookkeeping. This is intentionally separate from
// moe_ggml_mul_mat_id because the latter is also used by other model families
// and does not imply Qwen4Exp router semantics.
inline bool &moe_ggml_gpu_router_flag() { static bool v = false; return v; }
inline bool moe_ggml_gpu_router() {
  // Unlike the older feature flags, this one needs an explicit `=0` escape
  // hatch for performance A/B tests: the config setter may enable it for a
  // Qwen profile, while a run can still compare the existing AVX-512 host
  // router without making a temporary config copy.  The override is scoped
  // to this Qwen-only feature; other model routers do not consult it.
  static const int env_override = [] {
    const char *e = std::getenv("MOESTREAM_GGML_GPU_ROUTER");
    if (!e || !e[0]) return -1;
    return e[0] == '1' ? 1 : 0;
  }();
  return env_override >= 0 ? env_override != 0 : moe_ggml_gpu_router_flag();
}
inline void set_moe_ggml_gpu_router(bool v) { moe_ggml_gpu_router_flag() = v; }

// Which GPU the ggml MoE graph runs on. -1 (default) auto-selects the vault
// GPU with the most FREE VRAM -- which is NOT the same as the best choice: on
// a box whose primary vault card is deliberately filled with expert cache,
// "most free" resolves to the OTHER card, and here that is the PCIe x8 5060 Ti
// (6.9 GB/s H2D) rather than the x16 3090 (12.3 GB/s). MEASURED 2026-09-05 on
// qwen4exp: same run, GPU 1 -> 10.22s prefill, GPU 0 -> 9.24s. Set this
// explicitly (and leave the chosen card ~700 MB of headroom via its own
// moe_cache_gb) rather than relying on the automatic choice.
// MOESTREAM_GGML_MOE_DEVICE overrides both, for A/B without a config edit.
inline int& moe_ggml_device_flag() {
  static int v = -1;
  return v;
}
inline int moe_ggml_device() { return moe_ggml_device_flag(); }
inline void set_moe_ggml_device(int v) { moe_ggml_device_flag() = v; }

// --- hc_ggml (EngineConfig, default OFF) ---------------------------------
// Runs Qwen4Exp's hyper-connections as ggml graphs on the GPU instead of the
// CPU implementation in src/hyper_connections.cpp. MEASURED 2026-09-05 on a
// 441-token prefill: HC is 61% of the pass -- 3.29 s of CPU compute PLUS
// ~2.3 s of the dense_gpu bucket, because its three dense_proj calls push
// 18.1 MB `xn`/`gate` tensors across PCIe six times per layer (5.3 GB per
// prefill). Both costs only go away together, which needs the intermediates
// to stay in VRAM -- see include/ggml_hc_backend.hpp.
// Requires a build with MOESTREAM_USE_GGML_CUDA. Falls back to the CPU path
// for the whole forward pass if anything fails, never per bracket (a partial
// switch would leave the host and device copies of hc_stream_buf divergent).
inline bool& hc_ggml_flag() {
  static bool v = false;
  return v;
}
inline bool hc_ggml() {
  static const bool env = [] {
    const char *e = std::getenv("MOESTREAM_HC_GGML");
    return e && e[0] == '1';
  }();
  return env || hc_ggml_flag();
}
inline void set_hc_ggml(bool v) { hc_ggml_flag() = v; }

// --- gdn_proj_ggml (EngineConfig, default OFF) ---------------------------
// Fuses Qwen4Exp GDN's four shared-input dense_proj calls (attn_qkv,
// attn_gate, ssm_alpha, ssm_beta -- all four read the SAME hidden_states)
// into ONE ggml graph, so the activation is uploaded once instead of four
// times and the decay-log/beta-sigmoid postprocessing runs as graph ops
// instead of two more CPU loops needing their own round trip. See
// include/ggml_gdn_proj_backend.hpp for the measurement and the deliberately
// narrow scope (conv1d/L2-norm/delta-rule/output-norm are untouched -- they
// already have their own real fast paths).
// Requires a build with MOESTREAM_USE_GGML_CUDA. Falls back to the existing
// per-call dense_proj path for the whole call if anything fails.
inline bool& gdn_proj_ggml_flag() {
  static bool v = false;
  return v;
}
inline bool gdn_proj_ggml() {
  static const bool env = [] {
    const char *e = std::getenv("MOESTREAM_GDN_PROJ_GGML");
    return e && e[0] == '1';
  }();
  return env || gdn_proj_ggml_flag();
}
inline void set_gdn_proj_ggml(bool v) { gdn_proj_ggml_flag() = v; }

// Complete Qwen4Exp GDN on one GGML CUDA graph, including the recurrence,
// gated output norm and ssm_out projection. This is separate from the
// projection-only switch so it can be validated with an exact output A/B.
inline bool& gdn_full_ggml_flag() {
  static bool v = false;
  return v;
}
inline bool gdn_full_ggml() {
  static const bool env = [] {
    const char *e = std::getenv("MOESTREAM_GDN_FULL_GGML");
    return e && e[0] == '1';
  }();
  return env || gdn_full_ggml_flag();
}
inline void set_gdn_full_ggml(bool v) { gdn_full_ggml_flag() = v; }

// MOESTREAM_EXPERT_STREAM_NOSYNC=1 drops the per-expert
// cudaStreamSynchronize when moe_gpu_accum makes it unnecessary (the
// accumulate kernel is stream-ordered behind the down-proj, so nothing
// crosses back to the host). MEASURED 2026-09-05 as a REGRESSION on
// qwen4exp -- removing the sync lets the copy engine run continuously,
// and the staging memcpy then contends with it for DRAM (`disk=` tripled,
// 2.27s -> 6.05s, net prefill 20.44s -> 23.10s). Kept behind an env flag
// rather than deleted: the sync IS genuinely redundant, so this becomes a
// win the moment staging stops competing for the same bandwidth (e.g. if
// the mmap pages are ever DMA'd without a host-side copy).
// --- moe_expert_bf16_gemm (EngineConfig, default OFF) --------------------
// Routes the ROUTED-EXPERT gate/up/down matmuls through
// dequantise-to-BF16 + cuBLAS tensor-core GEMM instead of the fused
// GEMV-shaped kernel, for S>1 (prefill) only.
//
// Why it matters and why it looked worthless before: the fused K-quant
// kernel's grid is (O, S), so it RE-READS the whole weight row for every
// token row -- ~21x amplification at qwen4exp's real prefill shape. The
// BF16 path dequantises the weight once and lets cuBLAS reuse it across
// all rows. This was written in 2026-09-04 and measured as "no effect",
// but that measurement was invalid: the code sat between `switch (type) {`
// and the first `case`, i.e. it was unreachable and never ran (found and
// fixed 2026-09-05 -- see gpu_matmul_dispatch's own comment). With it
// actually reachable, per-expert GPU time fell 0.369 -> 0.133 ms.
//
// Default OFF because it is a real PRECISION change, not just a speed one:
// BF16 has ~8 mantissa bits, which is enough to flip a close greedy
// argmax, so generated text can differ from the fused path (verified
// coherent and on-topic, but not token-identical). MOESTREAM_EXPERT_BF16_
// GEMM=1 still forces it on for A/B without touching a config.
inline bool& moe_expert_bf16_gemm_flag() {
  static bool v = false;
  return v;
}
inline bool moe_expert_bf16_gemm() {
  static const bool env = [] {
    const char *e = std::getenv("MOESTREAM_EXPERT_BF16_GEMM");
    return e && e[0] == '1';
  }();
  return env || moe_expert_bf16_gemm_flag();
}
inline void set_moe_expert_bf16_gemm(bool v) { moe_expert_bf16_gemm_flag() = v; }

inline bool expert_stream_no_sync() {
  static const bool v = [] {
    const char *e = std::getenv("MOESTREAM_EXPERT_STREAM_NOSYNC");
    return e && *e == '1';
  }();
  return v;
}
inline void set_moe_prefill_stream_chunk(int v) {
  moe_prefill_stream_chunk_flag() = (v > 0 ? v : 1);
}

// dense_bf16_prefill_gemm's accessors live in kquant_dequant.hpp, NOT here:
// cuda_manager.hpp needs them and cannot include this header (dense_dispatch.hpp
// includes cuda_manager.hpp, so the dependency only goes one way). Same reason
// cpu_dot_ggml_avx2_enabled() lives over there.

// SwiGLU clamp SEMANTICS switch (GLM5Next / GLM 5.3 Flash). Off (default):
// this project's long-standing order -- silu(gate), multiply by up, then
// clamp the PRODUCT to +/-limit (swiglu_clamp_inplace, execution_layer.cpp).
// On: ggml_swiglu_clamp's exact fused semantics (llama.cpp ggml-cpu
// ops.cpp), which the reference uses for GLM5NEXT (and DEEPSEEK4):
//   gate = min(gate_raw, limit);  up = clamp(up_raw, +/-limit);
//   out  = silu(gate) * up       (the product itself is NOT clamped).
// These are genuinely different functions anywhere gate_raw > limit or
// |up_raw| > limit. Set once by graph_initializer.cpp when the model family
// is GLM5NEXT; read on the expert-FFN hot path with no locking (never
// mutated after startup), same discipline as cpu_only_flag() above.
// NOTE: the reference routes DEEPSEEK4 through the fused op too, but
// MoEStream's DeepSeek-V4 currently uses the product-clamp order (verified
// coherent on real hardware before this discrepancy was catalogued) --
// reconciling DSV4 is a separate, hardware-A/B-gated decision, deliberately
// not bundled into the GLM5Next bring-up.
inline bool& swiglu_clamp_fused_flag() {
    static bool v = false;
    return v;
}
inline bool swiglu_clamp_fused() { return swiglu_clamp_fused_flag(); }
inline void set_swiglu_clamp_fused(bool on) { swiglu_clamp_fused_flag() = on; }

// dsv4_outproj_gpu (config): enables cuBLAS GPU dispatch for DeepSeek-V4
// attention's grouped output projection (attn_output_a), whose weight gets
// a one-time persistent VRAM upload per layer -- see EngineConfig::
// dsv4_outproj_gpu's own comment for the full reasoning (memory-bandwidth-
// bound on CPU, ~12x faster once VRAM-resident, real ~5.8GB VRAM cost).
// Same function-local-static-storage pattern as cpu_only_flag() above --
// set once by main.cpp after config load, read from attention_layer.cpp's
// hot path with no locking needed since it's never mutated after startup.
inline bool& dsv4_outproj_gpu_flag() {
    static bool v = false;
    return v;
}
inline bool dsv4_outproj_gpu_enabled() { return dsv4_outproj_gpu_flag(); }
inline void enable_dsv4_outproj_gpu() { dsv4_outproj_gpu_flag() = true; }

// dsv4_score_gpu (config): enables the cuBLAS GPU path for DeepSeek-V4's
// raw-SWA and HCA attention score phase (QK^T + softmax + weighted-V over
// the raw ring-buffer window, plus HCA's dense compressed-block rows when
// present) -- see EngineConfig::dsv4_score_gpu's own comment for the full
// reasoning, and dsv4_score_gpu_batched_host (cuda_manager.hpp) for why CSA
// (ratio==4) layers are excluded. Same function-local-static-storage
// pattern as dsv4_outproj_gpu_flag() above -- set once by main.cpp after
// config load, read from attention_layer.cpp's hot path with no locking
// needed since it's never mutated after startup.
inline bool& dsv4_score_gpu_flag() {
    static bool v = false;
    return v;
}
inline bool dsv4_score_gpu_enabled() { return dsv4_score_gpu_flag(); }
inline void enable_dsv4_score_gpu() { dsv4_score_gpu_flag() = true; }

// cpu_dot_ggml_avx2_flag()/_enabled()/enable_cpu_dot_ggml_avx2(): defined
// in kquant_dequant.hpp, not here -- that header is cpu_quant_matmul's own
// home, and this file already #includes it above (kquant_dequant.hpp
// includes engine.hpp/kernels_cpu.hpp only, never this file, so no
// circular-include risk) -- so the real definitions are already reachable
// here by the time any code below this point calls them. See that
// header's own comment for the full flag semantics.

// Per-layer version of the same idea: `cpu_affinity_layers` (config) names
// specific MoE layers whose routed experts should NEVER be dispatched on
// GPU or promoted/uploaded into VRAM at all -- unlike cpu_only_mode() this
// is scoped to just those layers, everything else still dispatches
// normally. Motivated by layers whose expert set is so large/flat that a
// real run spends far more wall-clock time on the disk read than the
// matmul itself (see CLAUDE.md) -- for those layers, paying for a VRAM
// upload buys nothing (the bottleneck never moves) while it DOES cost real
// VRAM cache capacity taken away from layers where GPU residency actually
// helps, plus real PCIe/DMA time that's pure overhead if the compute was
// going to be I/O-bound either way. Backed by a flat bool vector (not a
// set) for an O(1), branch-predictable check on the hottest possible path
// (once per routed expert per token) -- set once at startup
// (set_cpu_affinity_layers, main.cpp, from EngineConfig::cpu_affinity_layers)
// before any dispatch can run, never mutated afterward, so no locking is
// needed despite being read from multiple threads (prefetcher, decode).
inline std::vector<bool>& cpu_affinity_layer_flags() {
    static std::vector<bool> v;
    return v;
}
inline void set_cpu_affinity_layers(const std::vector<int>& layers, int n_layers) {
    std::vector<bool> v(std::max(0, n_layers), false);
    for (int l : layers) {
        if (l >= 0 && l < (int)v.size()) v[(size_t)l] = true;
    }
    cpu_affinity_layer_flags() = std::move(v);
}
inline bool is_cpu_affinity_layer(int layer_idx) {
    const auto& v = cpu_affinity_layer_flags();
    return layer_idx >= 0 && layer_idx < (int)v.size() && v[(size_t)layer_idx];
}

// attn_score_threads (config): dedicated OMP team size for the causal
// attention score/softmax/value-aggregate loop (attention_layer.cpp's
// `#pragma omp parallel for collapse(2)` score computation) -- SEPARATE
// from the general omp_max_threads default. Motivated by a real-hardware
// finding (2026-08-13): score is the one attention sub-step that's 100%
// CPU (no GPU path exists for it, see CLAUDE.md), and raising expert_
// fetch_threads/omp_max_threads 8->10 for a real, measured disk-I/O win
// simultaneously raised score's own per-token cost ~74.8->83.5ms/token in
// the SAME real run -- the two regions now compete for the same 12
// physical cores. Same function-local-static-storage idiom as
// cpu_affinity_layer_flags() above; set once at startup (set_attn_score_
// threads, main.cpp, from EngineConfig::attn_score_threads), read from
// attention_layer.cpp's hot path with no locking needed.
//
// 0 (the flag's own default) means "no override" -- attn_score_threads()
// falls back to omp_get_max_threads(), i.e. whatever the process-wide
// default (omp_max_threads, or its own auto value) already is, so a
// config that never sets this behaves identically to before this existed.
// A caller explicitly passing 0 or a negative number to num_threads()
// would be undefined behavior, which is exactly why this function -- not
// the raw config field -- is what call sites should use.
inline int& attn_score_threads_flag() {
    static int v = 0;
    return v;
}
inline void set_attn_score_threads(int n) { attn_score_threads_flag() = n; }
inline int attn_score_threads() {
    int v = attn_score_threads_flag();
    return v > 0 ? v : omp_get_max_threads();
}

// Per-layer override for the pilot router's prefetch hint count
// (EngineConfig::prefetch_pilot_k / prefetch_pilot_k_per_layer). Same
// set-once-at-startup, read-from-multiple-threads-without-locking idiom as
// cpu_affinity_layer_flags() above (set_prefetch_pilot_k, main.cpp; read
// from execution_layer.cpp's pilot loop, decode thread only in practice but
// no different in principle from the cpu_affinity flags' own read pattern).
// A missing or negative per-layer entry means "no override for this
// layer" -- falls back to the global default, not to 0 -- distinguishing
// "not configured" from "the user explicitly asked for 0 hints here".
inline std::vector<int>& pilot_k_per_layer_overrides() {
    static std::vector<int> v;
    return v;
}
inline int& pilot_k_default_value() {
    static int v = 8;
    return v;
}
inline void set_prefetch_pilot_k(const std::vector<int>& per_layer, int default_k) {
    pilot_k_per_layer_overrides() = per_layer;
    pilot_k_default_value() = default_k;
}
inline int pilot_k_for_layer(int layer_idx) {
    const auto& v = pilot_k_per_layer_overrides();
    if (layer_idx >= 0 && layer_idx < (int)v.size() && v[(size_t)layer_idx] >= 0) return v[(size_t)layer_idx];
    return pilot_k_default_value();
}

// prefetch_pilot_k_adaptive (2026-09-13): "most confident numbers" fetch
// width -- see EngineConfig::prefetch_pilot_k_adaptive's own comment for
// the full motivation (22.8% of real prefetches this session were pure
// LRU-evicted-before-use waste, concentrated in the low-confidence tail
// ranks that also happen to be the MAJORITY of the fetch-worthy volume).
// Same set-once-at-startup idiom as pilot_k_for_layer above.
struct PilotKAdaptiveConfig {
    bool enabled = false;
    double threshold = 0.50;
    int min_samples = 30;
};
inline PilotKAdaptiveConfig& pilot_k_adaptive_config() {
    static PilotKAdaptiveConfig v;
    return v;
}
inline void set_pilot_k_adaptive_config(bool enabled, double threshold, int min_samples) {
    auto& v = pilot_k_adaptive_config();
    v.enabled = enabled;
    v.threshold = threshold;
    v.min_samples = min_samples;
}
// Returns the effective fetch width for this layer transition: the
// existing static ceiling (pilot_k_for_layer, ALREADY min'd against
// c.topk by the caller) when the mode is off, cold-start (rank 0 itself
// lacks min_samples live checks yet), or out of the tracked-layer range
// -- same "no real signal yet means don't act on it" shape as every other
// gate in this file. Otherwise walks ranks 0..ceiling-1, using the SAME
// live per-layer-rank counters [PILOT-BY-LAYER-RANK] reports
// (PerfCounters::n_pilot_recall_hits_by_layer_rank/checks_by_layer_rank
// -- marginal, all-checks accuracy, not fetch-worthy-only; a real,
// already-collected proxy for "is predicting this rank at this layer
// generally reliable", one gate reuse cheaper than adding a second,
// fetch-worthy-only 2D counter), stopping at the first rank whose own
// accuracy drops below threshold (contiguous-prefix, matches ram_vram_
// promote_adaptive's own proven shape) -- so a layer never fetches past
// the point its own live data says is more likely wrong than right.
// Deliberately floor 1 (always fetch at least the single top prediction)
// and ceiling-capped (never fetches MORE than the existing static width
// would have) -- narrows only, never widens.
inline int adaptive_pilot_k_for_layer(int layer_idx, int ceiling) {
    const auto& cfg = pilot_k_adaptive_config();
    if (!cfg.enabled) return ceiling;
    if (layer_idx < 0 || layer_idx >= PerfCounters::kMaxTrackedLayersForMiss) return ceiling;
    if (ceiling <= 1) return ceiling;
    auto& p = perf();
    int kept = 0;
    int cap = std::min(ceiling, 8);  // n_pilot_recall_*_by_layer_rank is a fixed [8] array
    for (int k = 0; k < cap; ++k) {
        uint32_t checks = p.n_pilot_recall_checks_by_layer_rank[layer_idx][k].load(std::memory_order_relaxed);
        if (checks < (uint32_t)cfg.min_samples) {
            // No real signal yet for this rank -- if this is rank 0 (cold
            // start for the whole layer), fall back to the full ceiling
            // unchanged; otherwise stop extending here (can't trust an
            // unmeasured rank enough to include it, but ranks already
            // accepted below it stay accepted).
            return k == 0 ? ceiling : std::max(1, kept);
        }
        uint32_t hits = p.n_pilot_recall_hits_by_layer_rank[layer_idx][k].load(std::memory_order_relaxed);
        double acc = (double)hits / (double)checks;
        if (acc < cfg.threshold) return std::max(1, kept);
        kept = k + 1;
    }
    return std::max(1, kept);
}

// Same set-once-at-startup / read-everywhere idiom as pilot_k_for_layer
// above, for EngineConfig::prefetch_pilot_depth / prefetch_pilot_k_depth2
// (set_prefetch_pilot_depth, main.cpp; read from execution_layer.cpp's
// pilot loop). No per-layer override for depth-2's k yet (unlike l+1's
// pilot_k_per_layer_overrides) -- pilot_k2_for_layer keeps the same
// per-layer function SHAPE anyway, so one can be added later without
// changing every call site, mirroring pilot_k_for_layer's own signature.
inline int& pilot_depth_value() {
    static int v = 1;
    return v;
}
inline int& pilot_k2_default_value() {
    static int v = 8;
    return v;
}
inline void set_prefetch_pilot_depth(int depth, int k2_default) {
    pilot_depth_value() = depth;
    pilot_k2_default_value() = k2_default;
}
// Whether the pilot should predict+push real depth-2 (l+2) hints, as
// opposed to depth-2 measurement continuing to run for free either way
// (see execution_layer.cpp's pilot loop -- the recall counters are always
// live regardless of this).
inline bool prefetch_pilot_depth2_enabled() {
    return pilot_depth_value() == 2;
}
inline int pilot_k2_for_layer(int layer_idx) {
    (void)layer_idx;  // no per-layer override yet -- see this block's own comment
    return pilot_k2_default_value();
}

// ATTN-PILOT real-hint wiring (2026-08-25) -- EngineConfig::
// prefetch_attn_pilot/prefetch_attn_pilot_k, same set-once/read-everywhere
// idiom as the pair above. See execution_layer.cpp's ATTN-PILOT firing
// site (inside the same block DEPTH-1 uses) and CLAUDE.md's "CPU-offload
// feasibility"-adjacent pilot investigation for the real recall data this
// was sized against (75.8% overall vs DEPTH-1's 84.8% for the SAME target
// layer, or vs DEPTH-2's 72.1% when read as reaching 2 layers past the
// last known-complete layer).
inline bool& attn_pilot_enabled_flag() {
    static bool v = false;
    return v;
}
inline int& attn_pilot_k_default_value() {
    static int v = 8;
    return v;
}
inline void set_attn_pilot_enabled(bool enabled, int k_default) {
    attn_pilot_enabled_flag() = enabled;
    attn_pilot_k_default_value() = k_default;
}
inline bool prefetch_attn_pilot_enabled() { return attn_pilot_enabled_flag(); }
inline int attn_pilot_k_for_layer(int layer_idx) {
    (void)layer_idx;  // no per-layer override -- mirrors pilot_k2_for_layer's own shape
    return attn_pilot_k_default_value();
}

// attn_pilot_vram_promote_top_k (2026-08-25): SEPARATE from prefetch_attn_
// pilot_k above -- that governs plain RAM-tier real-hint pushing (push_
// hint); this governs an ADDITIONAL, independent decision: of ATTN-PILOT's
// own already_ram-classified picks (RAM-resident, not yet VRAM-resident,
// regardless of which mechanism put them in RAM), how many of the
// top-ranked ones get speculatively promoted to VRAM via RAMSafetyManager::
// promote_ram_to_vram -- the SAME proven, COPY-only (allow_move=false)
// mechanism DEPTH-1's own pilot loop already uses, NOT a new promotion
// pathway. Deliberately a SEPARATE cutoff from the shared ram_vram_
// promote_top_k (DEPTH-1's own, settled at 2) rather than reusing it --
// ATTN-PILOT's own accuracy/call-volume profile is different, so that
// already-tuned value has no reason to also be right here. 0 (default) =
// off. Real per-rank ATTN-PILOT accuracy (see CLAUDE.md's pilot
// investigation) should inform this before raising it far past a small
// value -- the SAME real regression `ram_vram_promote_top_k=3` caused for
// DEPTH-1 (over-aggressive speculative promotion -> VRAM churn -> lower
// cache-hit rate) is the exact risk class here too.
inline int& attn_pilot_vram_promote_top_k_value() {
    static int v = 0;
    return v;
}
inline void set_attn_pilot_vram_promote_top_k(int top_k) {
    attn_pilot_vram_promote_top_k_value() = top_k;
}

// ram_vram_promote_top_k / ram_vram_promote_adaptive (config): pilot-
// prediction-driven speculative RAM->VRAM promotion, calibrated against the
// "ram-only" per-rank metric (PerfCounters::n_pilot_ram_only_hits_by_rank /
// n_pilot_ram_only_checks_by_rank) -- see EngineConfig's own comment for
// the full design and CLAUDE.md for the real-hardware data this was
// calibrated against. Same set-once-at-startup / read-from-decode-thread
// idiom as pilot_k_for_layer above (set_ram_vram_promote_config, main.cpp;
// read from execution_layer.cpp's l+1 pilot loop, the "already_ram" branch
// specifically -- see RAMSafetyManager::promote_ram_to_vram's own comment
// for why that's the only place this may be called from). Mutual
// exclusivity between the two modes (at most one of top_k>0 / adaptive) is
// already enforced by EngineConfig's own validation before this is ever
// set -- not re-checked here.
struct RamVramPromoteConfig {
    int top_k = 0;
    bool adaptive = false;
    double adaptive_threshold = 0.30;
    int min_samples = 30;
    // dynamic_layer (2026-08-26): a third, mutually-exclusive mode -- see
    // EngineConfig::ram_vram_promote_dynamic_layer's own comment for the
    // full motivation (direct graduation of ram_vram_promote_layer_gate
    // from a plain veto to a smooth per-layer width) and for why this is
    // FIVE explicit tiers (widened from an original 3-tier min/base/max
    // design once real [PILOT-BY-LAYER] data showed a genuine, narrow
    // >=90%-accuracy population worth a bigger bump than the already-
    // validated 80-90% band). Tier widths/thresholds read by
    // dynamic_layer_top_k below, not by this struct itself -- kept here
    // purely as the config record, same role top_k/adaptive_threshold play
    // for their own modes.
    bool dynamic_layer = false;
    double dynamic_threshold_4 = 0.90;
    double dynamic_threshold_3 = 0.80;
    double dynamic_threshold_2 = 0.70;
    double dynamic_threshold_1 = 0.50;
    int dynamic_k_tier4 = 4;
    int dynamic_k_tier3 = 3;
    int dynamic_k_tier2 = 2;
    int dynamic_k_tier1 = 1;
    int dynamic_k_floor = 0;
    int dynamic_cold_start_k = 2;
    int dynamic_min_samples = 30;
};
inline RamVramPromoteConfig& ram_vram_promote_config() {
    static RamVramPromoteConfig v;
    return v;
}
inline void set_ram_vram_promote_config(int top_k, bool adaptive, double adaptive_threshold, int min_samples) {
    auto& v = ram_vram_promote_config();
    v.top_k = top_k;
    v.adaptive = adaptive;
    v.adaptive_threshold = adaptive_threshold;
    v.min_samples = min_samples;
}
inline void set_ram_vram_promote_dynamic_config(bool enabled,
                                                 double threshold_4, double threshold_3,
                                                 double threshold_2, double threshold_1,
                                                 int k_tier4, int k_tier3, int k_tier2, int k_tier1,
                                                 int k_floor, int cold_start_k, int min_samples) {
    auto& v = ram_vram_promote_config();
    v.dynamic_layer = enabled;
    v.dynamic_threshold_4 = threshold_4;
    v.dynamic_threshold_3 = threshold_3;
    v.dynamic_threshold_2 = threshold_2;
    v.dynamic_threshold_1 = threshold_1;
    v.dynamic_k_tier4 = k_tier4;
    v.dynamic_k_tier3 = k_tier3;
    v.dynamic_k_tier2 = k_tier2;
    v.dynamic_k_tier1 = k_tier1;
    v.dynamic_k_floor = k_floor;
    v.dynamic_cold_start_k = cold_start_k;
    v.dynamic_min_samples = min_samples;
}
// attn_pilot_vram_promote_dynamic (2026-08-26): opts ATTN-PILOT's own
// promotion into the SAME dynamic_layer tiers above (RamVramPromoteConfig::
// dynamic_*), judged against ATTN-PILOT's own per-layer counters via
// dynamic_layer_top_k's is_attn_pilot parameter -- see EngineConfig::
// attn_pilot_vram_promote_dynamic's own comment. Only has any effect when
// ram_vram_promote_config().dynamic_layer is ALSO true.
inline bool& attn_pilot_vram_promote_dynamic_flag() {
    static bool v = false;
    return v;
}
inline void set_attn_pilot_vram_promote_dynamic(bool enabled) {
    attn_pilot_vram_promote_dynamic_flag() = enabled;
}
inline bool attn_pilot_vram_promote_dynamic_enabled() { return attn_pilot_vram_promote_dynamic_flag(); }

// speculative_promote_to_active_layer (2026-08-26): when true, execution_
// layer.cpp's pilot promotion call sites route through RAMSafetyManager::
// promote_ram_to_active_layer instead of promote_ram_to_vram -- see
// EngineConfig::speculative_promote_to_active_layer's own comment for the
// full motivation. Same function-local-static-storage accessor pattern as
// every other simple process-wide flag in this file (cpu_only_mode,
// attn_pilot_vram_promote_dynamic_flag above).
inline bool& speculative_promote_to_active_layer_flag() {
    static bool v = false;
    return v;
}
inline void set_speculative_promote_to_active_layer(bool enabled) {
    speculative_promote_to_active_layer_flag() = enabled;
}
inline bool speculative_promote_to_active_layer_enabled() { return speculative_promote_to_active_layer_flag(); }

// speculative_active_layer_max_k (2026-08-26, EngineConfig::speculative_
// active_layer_max_k, default 2): a SEPARATE, narrower rank cutoff for the
// active-layer redirect specifically, independent of whatever width
// ram_vram_promote_top_k/dynamic_layer already decided for the (now
// bypassed, when the flag above is on) main-cache path. Real A/B (see
// CLAUDE.md) found the redirect mechanically correct (cache hit up,
// waste low, main-cache eviction fully eliminated, synchronous-blocking
// bug fixed via batched async) but still net-negative on decode
// throughput at the full dynamic-tier width (up to k=3), because the
// added DMA volume competes for the SAME already bandwidth-bound active-
// layer transfer pipe decode_async_prewarm also uses (this project's own
// single largest, most-investigated decode bucket). This exists to trade
// coverage for volume -- narrower cutoff, same mechanism -- exactly the
// lesson every prior over-promotion-volume investigation in this project
// already reached. Applied as an ADDITIONAL min() on top of whatever
// should_promote/attn_promote_cutoff already gated, never a widening.
inline int& speculative_active_layer_max_k_value() {
    static int v = 2;
    return v;
}
inline void set_speculative_active_layer_max_k(int max_k) {
    speculative_active_layer_max_k_value() = max_k;
}

// speculative_active_layer_promote_on_confirm (2026-08-27, EngineConfig::
// speculative_active_layer_promote_on_confirm, default false -- preserves
// existing behavior exactly unless explicitly opted in): real hardware
// data (a matched V1/V2 [REQUEST] netting, CLAUDE.md's 2026-08-27 entry)
// found the active-layer redirect isn't a pure reroute of the SAME real
// work -- it's a reroute that also loses persistence, because upload_
// expert_active_async unconditionally sets prewarm_skip_next_promote=true
// for a speculative fill too (find_active_expert.cpp), same as it does
// for decode_async_prewarm's own GUARANTEED same-token touch. That's
// correct for decode_async_prewarm (its first touch is 100% certain and
// carries zero reuse signal) but NOT for a speculative fill (~94-95%
// correct, not certain -- the touch genuinely confirms the guess). Net
// effect: since active_layer_promote_hit_threshold=1 needs the SECOND
// real touch to promote, and a speculative fill's active-layer window
// for that layer index is gone well before a LATER decode token could
// ever revisit it, promotion is effectively impossible for this path
// (Three-tier caching's own promoted-to-main count sits at ~9-10 out of
// ~3000 speculative fills, real-hardware confirmed) -- so a correctly-
// predicted expert gets re-fetched from scratch on every later token
// that needs it again. Directly explains a real, measured 30-39% MORE
// promotion-attempt volume in V2 vs. V1's own persistent main-cache
// promotion count for the same real underlying need (2371->3079 and
// 2143->2982 across two matched A/B pairs). When true, a
// speculative fill's first CONFIRMING touch is NOT skipped -- it counts
// toward hit_count immediately, promoting to the real main cache on the
// first genuine use (matching V1's own "promote once, benefit forever"
// property) instead of staying permanently ephemeral. Known, real risk,
// gated behind its own flag rather than changed as default behavior:
// this reopens the ORIGINAL problem speculative_promote_to_active_layer
// was built to avoid in the first place (main-cache eviction damage from
// speculative guesses) -- just at a much lower wrong-guess rate (~5-6%
// vs. whatever the original main-cache-direct design saw).
//
// TESTED ON REAL HARDWARE (2026-08-27), a clean decisive NEGATIVE
// result -- the risk above wasn't hypothetical. promoted-to-main jumped
// from ~10 (baseline) to 2385-2542 (~240x, matching the real ~94-95%
// confirmation rate almost exactly -- the mechanism works as designed),
// but decode dropped 0.88->0.76-0.77 (13-14% regression) and one of two
// runs threw a real CUDA error ("Failed to async-upload expert to
// active layer: invalid argument" / "cudaMemcpy H2D failed") -- a
// genuine stability issue, not just a slowdown. This is now the THIRD
// real regression from this exact class of promotion-threshold change
// in this codebase (see the two Incident writeups in CLAUDE.md). Leave
// off; don't re-enable without first hardening whatever the upload path
// hit under this volume, and expect a real throughput cost even then.
inline bool& speculative_active_layer_promote_on_confirm_flag() {
    static bool v = false;
    return v;
}
inline void set_speculative_active_layer_promote_on_confirm(bool enabled) {
    speculative_active_layer_promote_on_confirm_flag() = enabled;
}
inline bool speculative_active_layer_promote_on_confirm_enabled() {
    return speculative_active_layer_promote_on_confirm_flag();
}

// dynamic_layer_top_k (2026-08-26): returns the effective promotion width
// for (layer_idx, is_attn_pilot) under RamVramPromoteConfig::dynamic_layer
// mode -- see that field's own comment (EngineConfig) for the exact
// five-tier rule. Falls back to dynamic_cold_start_k whenever there isn't
// yet real signal to differentiate on (layer index out of tracked range,
// fewer than dynamic_min_samples live checks, or zero topk_sum) --
// deliberately a SEPARATE field from dynamic_k_tier2 even though they
// default to the same value, so cold-start behavior doesn't silently
// change if k_tier2 is ever re-tuned on its own. Reads the exact same live
// per-layer counters layer_promote_allowed does (and [PILOT-BY-LAYER]
// reports) -- always consistent with what that report would show if
// printed right now. Tier checks walk from the highest threshold down --
// each band is genuinely "in [threshold_N, threshold_{N+1})", implemented
// as an early-return once accuracy clears a given threshold rather than an
// explicit range check.
inline int dynamic_layer_top_k(int layer_idx, bool is_attn_pilot) {
    const auto& cfg = ram_vram_promote_config();
    if (layer_idx < 0 || layer_idx >= PerfCounters::kMaxTrackedLayersForMiss) return cfg.dynamic_cold_start_k;
    auto& p = perf();
    uint32_t checks = is_attn_pilot
        ? p.n_pilot_attn_recall_checks_by_layer[layer_idx].load(std::memory_order_relaxed)
        : p.n_pilot_recall_checks_by_layer[layer_idx].load(std::memory_order_relaxed);
    if (checks < (uint32_t)cfg.dynamic_min_samples) return cfg.dynamic_cold_start_k;
    uint32_t hits = is_attn_pilot
        ? p.n_pilot_attn_recall_hits_by_layer[layer_idx].load(std::memory_order_relaxed)
        : p.n_pilot_recall_hits_by_layer[layer_idx].load(std::memory_order_relaxed);
    uint32_t topk_sum = is_attn_pilot
        ? p.n_pilot_attn_recall_topk_sum_by_layer[layer_idx].load(std::memory_order_relaxed)
        : p.n_pilot_recall_topk_sum_by_layer[layer_idx].load(std::memory_order_relaxed);
    if (topk_sum == 0) return cfg.dynamic_cold_start_k;
    double acc = (double)hits / (double)topk_sum;
    if (acc >= cfg.dynamic_threshold_4) return cfg.dynamic_k_tier4;
    if (acc >= cfg.dynamic_threshold_3) return cfg.dynamic_k_tier3;
    if (acc >= cfg.dynamic_threshold_2) return cfg.dynamic_k_tier2;
    if (acc >= cfg.dynamic_threshold_1) return cfg.dynamic_k_tier1;
    return cfg.dynamic_k_floor;
}

// ===== Multi-GPU active-layer pool routing (2026-08-28) =====
// See EngineConfig::active_layer_routing_policy's own comment
// (engine_config.hpp) for the full real-hardware motivation. This is
// deliberately a small, swappable POLICY FUNCTION behind one enum switch,
// not logic inlined at each call site -- adding a new policy later means
// one new case here, zero changes to any caller.

// Minimal per-GPU view the router needs -- deliberately NOT GpuTopology::
// ActiveLayerGpuSlice itself, to avoid a circular include between this
// header and gpu_topology.hpp (this file is included BY gpu_topology-
// adjacent code, not the other way around).
struct ActiveLayerRouteGpu { int gpu_id; int weight; };

enum class ActiveLayerRoutingPolicy { Tiered, Proportional };

inline ActiveLayerRoutingPolicy parse_active_layer_routing_policy(const std::string& s) {
    if (s == "proportional") return ActiveLayerRoutingPolicy::Proportional;
    return ActiveLayerRoutingPolicy::Tiered;  // default, and any unrecognized string
}

struct ActiveLayerRoutingConfig {
    std::string policy_str = "tiered";
    // Sorted descending by weight (the setter guarantees this) -- empty
    // means "no multi-GPU routing configured", the byte-identical-to-
    // legacy default state.
    std::vector<ActiveLayerRouteGpu> gpus_sorted_desc;
};
inline ActiveLayerRoutingConfig& active_layer_routing_config() {
    static ActiveLayerRoutingConfig v;
    return v;
}
inline void set_active_layer_routing_config(const std::string& policy,
                                             const std::vector<ActiveLayerRouteGpu>& gpus) {
    auto& v = active_layer_routing_config();
    v.policy_str = policy;
    v.gpus_sorted_desc = gpus;
    std::sort(v.gpus_sorted_desc.begin(), v.gpus_sorted_desc.end(),
              [](const ActiveLayerRouteGpu& a, const ActiveLayerRouteGpu& b) { return a.weight > b.weight; });
}
// Only 2+ participating GPUs makes routing meaningful -- 0 or 1 means
// every caller should skip the routing decision entirely and use the
// legacy single-GPU call path (target_gpu_id = -1), not just get gpu_id
// back from a degenerate 1-entry router.
inline bool active_layer_multi_gpu_enabled() {
    return active_layer_routing_config().gpus_sorted_desc.size() >= 2;
}

// Fixed nominal rank range this router reasons about -- matches the
// architectural pilot rank width used pervasively elsewhere in this
// codebase (pilot_pred_ids/n_pilot_ram_only_hits_by_rank/etc. are all
// sized [8]). Using this FIXED denominator (rather than "however many
// candidates this specific transition actually promotes") keeps a given
// rank's tier assignment STABLE and predictable across transitions/tokens
// -- rank 6/7 always mean "the least confident possible pick" and always
// route the same way, regardless of whether a specific transition even
// reaches that rank. Avoids restructuring any caller's k-loop into a
// two-pass (count-then-route) shape just to know a real total upfront.
constexpr int ACTIVE_LAYER_ROUTING_NOMINAL_RANKS = 8;

// Pure routing logic (no global-state reads) -- kept separate from
// route_active_layer_gpu below specifically so it's independently
// testable without going through the live registry. gpus_sorted_desc
// must already be sorted descending by weight. rank is 0-based, already
// in PRIORITY order for this candidate stream (current-layer misses
// first, then DEPTH-1 rank 0,1,2..., then ATTN-PILOT -- whichever
// ordering the caller's own k-loop already uses). Returns -1 only if
// gpus_sorted_desc is empty (caller should treat as "no decision, use
// the legacy single-GPU path").
inline int route_active_layer_gpu_impl(ActiveLayerRoutingPolicy policy,
                                        const std::vector<ActiveLayerRouteGpu>& gpus_sorted_desc,
                                        int rank, int total_candidates) {
    if (gpus_sorted_desc.empty()) return -1;
    if (gpus_sorted_desc.size() == 1 || total_candidates <= 0) return gpus_sorted_desc[0].gpu_id;
    if (rank < 0) rank = 0;

    if (policy == ActiveLayerRoutingPolicy::Proportional) {
        // Deterministic proportional interleave: maps this candidate's
        // position in the (rank/total) unit interval to whichever GPU
        // owns that slice of the cumulative weight distribution. Achieves
        // roughly the configured ratio, spread across the whole sequence
        // rather than tail-concentrated on any one GPU -- unlike Tiered,
        // a high-priority rank CAN land on a low-weight GPU if its turn
        // falls there. No cross-call state needed (unlike a classic
        // stateful WRR scheduler), since rank/total already fully
        // determine the position.
        int total_weight = 0;
        for (auto& g : gpus_sorted_desc) total_weight += g.weight;
        if (total_weight <= 0) return gpus_sorted_desc[0].gpu_id;
        double pos = ((double)rank + 0.5) / (double)total_candidates;
        double cum = 0.0;
        for (auto& g : gpus_sorted_desc) {
            cum += (double)g.weight / (double)total_weight;
            if (pos < cum) return g.gpu_id;
        }
        return gpus_sorted_desc.back().gpu_id;
    }

    // Tiered (default): the weakest weight VALUE present defines the weak
    // tier (may be more than one GPU if several happen to share the
    // minimum weight); every other GPU is the strong tier. The weak tier
    // gets the LOWEST-priority TAIL of this candidate stream, sized
    // proportional to its own combined weight share of the total -- this
    // is what keeps risky/low-confidence picks off the weaker link,
    // matching the explicit design goal (route the stuff that can afford
    // to be slow or wrong to the weaker GPU). Within a tier, GPUs
    // round-robin evenly regardless of their own (possibly still-unequal)
    // weights -- a real, deliberate simplification for more than 2
    // distinct weight values (e.g. 3 GPUs all genuinely different
    // strengths); good enough for the common "N equal strong + 1 weak"
    // case this was built for (see EngineConfig::active_layer_routing_
    // policy's own comment), not claimed optimal beyond it.
    int min_w = gpus_sorted_desc[0].weight;
    for (auto& g : gpus_sorted_desc) min_w = std::min(min_w, g.weight);
    std::vector<int> weak_ids, strong_ids;
    int weak_weight_sum = 0, total_weight = 0;
    for (auto& g : gpus_sorted_desc) {
        total_weight += g.weight;
        if (g.weight <= min_w) { weak_ids.push_back(g.gpu_id); weak_weight_sum += g.weight; }
        else strong_ids.push_back(g.gpu_id);
    }
    if (strong_ids.empty()) {
        // Every configured GPU shares the same weight -- no real "weak
        // tier" to speak of (degenerate case); just round-robin everyone.
        return gpus_sorted_desc[(size_t)rank % gpus_sorted_desc.size()].gpu_id;
    }
    int weak_tail_size = (int)std::lround((double)total_candidates * (double)weak_weight_sum / (double)total_weight);
    if (weak_tail_size > total_candidates) weak_tail_size = total_candidates;
    int tail_start = total_candidates - weak_tail_size;
    if (rank >= tail_start) {
        return weak_ids[(size_t)(rank - tail_start) % weak_ids.size()];
    }
    return strong_ids[(size_t)rank % strong_ids.size()];
}

// Live call-site API -- reads the registry populated once at startup by
// set_active_layer_routing_config (main.cpp). Returns -1 (meaning "use the
// legacy single-GPU path, target_gpu_id=-1") whenever fewer than 2 GPUs
// are configured -- callers should check active_layer_multi_gpu_enabled()
// first (cheaper, avoids the call entirely) but this is safe to call
// unconditionally too.
//
// 2026-08-28 CORRECTION: total_candidates is now a REQUIRED parameter, no
// longer silently defaulted to ACTIVE_LAYER_ROUTING_NOMINAL_RANKS (8) --
// real hardware showed that fixed nominal denominator made Tiered's own
// tail genuinely unreachable in practice: the speculative-fill caller's
// own k never exceeds speculative_active_layer_max_k (4 by default), and
// decode_async_prewarm's own real candidate count per transition is
// typically well under 8 too (most of a token's top-8 picks are already
// VRAM-resident on a well-cached run) -- so a tail defined as "rank >= 6"
// was structurally unreachable from either real caller, routing ~1% of
// traffic instead of the intended weight ratio. Callers must now pass
// their own REAL candidate count for this specific transition/call (e.g.
// speculative_active_layer_max_k_value() for the speculative-fill path,
// decode_async_candidates.size() for decode_async_prewarm) so the tail is
// computed against what's actually reachable, not an architectural
// constant unrelated to either caller's own real range.
inline int route_active_layer_gpu(int rank, int total_candidates) {
    auto& cfg = active_layer_routing_config();
    if (cfg.gpus_sorted_desc.size() < 2) return -1;
    return route_active_layer_gpu_impl(parse_active_layer_routing_policy(cfg.policy_str),
                                        cfg.gpus_sorted_desc, rank, total_candidates);
}

// ram_vram_promote_layer_gate (2026-08-26): an ADDITIONAL veto layered on
// top of whichever mode above (or attn_pilot_vram_promote_top_k_value())
// already selected a promotion candidate -- see EngineConfig::ram_vram_
// promote_layer_gate's own comment for the full motivation (a real,
// large, per-LAYER accuracy spread the per-rank adaptive mode above
// can't see -- CLAUDE.md's [PILOT-BY-LAYER] entry). Same set-once-at-
// startup / read-from-decode-thread idiom as ram_vram_promote_config
// above (set_layer_promote_gate_config, main.cpp; read from execution_
// layer.cpp's l+1 pilot loop, both the DEPTH-1 and ATTN-PILOT promotion
// sites). layer_promote_allowed reads PerfCounters::n_pilot_recall_*_by_
// layer / n_pilot_attn_recall_*_by_layer directly (perf_timer.hpp, via
// engine.hpp -- already transitively included above) -- the SAME live
// counters [PILOT-BY-LAYER] itself reports, so this gate's decision is
// always consistent with what that report line would show if printed
// right now.
struct LayerPromoteGateConfig {
    bool enabled = false;
    double threshold = 0.65;
    int min_samples = 30;
    // top_fraction (2026-09-13): rank-based "most confident layers only"
    // mode -- see layer_promote_allowed's own comment below and
    // EngineConfig::ram_vram_promote_layer_gate_top_fraction's comment
    // for the full motivation. 0.0 = disabled, falls back to the flat
    // `threshold` above unchanged.
    double top_fraction = 0.0;
};
inline LayerPromoteGateConfig& layer_promote_gate_config() {
    static LayerPromoteGateConfig v;
    return v;
}
inline void set_layer_promote_gate_config(bool enabled, double threshold, int min_samples,
                                           double top_fraction = 0.0) {
    auto& v = layer_promote_gate_config();
    v.enabled = enabled;
    v.threshold = threshold;
    v.min_samples = min_samples;
    v.top_fraction = top_fraction;
}

// rank_based_layer_accuracy_cutoff (2026-09-13): backs layer_promote_
// allowed's top_fraction mode. Sorts every currently-tracked layer's OWN
// live accuracy (same counters [PILOT-BY-LAYER] reports) and returns the
// accuracy value at the top_fraction cut point -- callers then require
// acc >= this cutoff, which is a real "top N%" selection instead of an
// absolute pass/fail line. Deliberately CACHED and refreshed only every
// kRefreshInterval calls rather than recomputed on every call: this is
// reached from the decode-thread hot path (up to prefetch_pilot_k times
// per layer transition), and per-layer accuracy moves slowly over a
// session, so a coarse refresh is both safe and far cheaper than sorting
// ~kMaxTrackedLayersForMiss layers on every rank check. is_attn_pilot
// keeps DEPTH-1 and ATTN-PILOT ranked independently, matching every
// other per-pilot-type split in this gate. Returns 0.0 (== "allow all",
// same cold-start-safe default as the flat-threshold path) until the
// first layer anywhere has crossed min_samples -- no real signal yet
// means don't act on it, not "assume the worst".
inline double rank_based_layer_accuracy_cutoff(bool is_attn_pilot,
                                                const LayerPromoteGateConfig& cfg) {
    struct Cache {
        std::atomic<uint64_t> calls{0};
        std::atomic<double> cutoff{0.0};
    };
    static Cache depth1_cache;
    static Cache attn_cache;
    Cache& c = is_attn_pilot ? attn_cache : depth1_cache;

    constexpr uint64_t kRefreshInterval = 256;
    uint64_t n = c.calls.fetch_add(1, std::memory_order_relaxed);
    if (n % kRefreshInterval == 0) {
        auto& p = perf();
        std::vector<double> accs;
        accs.reserve(PerfCounters::kMaxTrackedLayersForMiss);
        for (int li = 0; li < PerfCounters::kMaxTrackedLayersForMiss; ++li) {
            uint32_t checks = is_attn_pilot
                ? p.n_pilot_attn_recall_checks_by_layer[li].load(std::memory_order_relaxed)
                : p.n_pilot_recall_checks_by_layer[li].load(std::memory_order_relaxed);
            if (checks < (uint32_t)cfg.min_samples) continue;
            uint32_t topk_sum = is_attn_pilot
                ? p.n_pilot_attn_recall_topk_sum_by_layer[li].load(std::memory_order_relaxed)
                : p.n_pilot_recall_topk_sum_by_layer[li].load(std::memory_order_relaxed);
            if (topk_sum == 0) continue;
            uint32_t hits = is_attn_pilot
                ? p.n_pilot_attn_recall_hits_by_layer[li].load(std::memory_order_relaxed)
                : p.n_pilot_recall_hits_by_layer[li].load(std::memory_order_relaxed);
            accs.push_back((double)hits / (double)topk_sum);
        }
        if (!accs.empty()) {
            std::sort(accs.begin(), accs.end(), std::greater<double>());
            size_t keep = (size_t)(accs.size() * cfg.top_fraction);
            if (keep == 0) keep = 1;  // always keep at least the single best layer
            if (keep > accs.size()) keep = accs.size();
            c.cutoff.store(accs[keep - 1], std::memory_order_relaxed);
        }
        // else: leave the cached cutoff as-is (0.0 on the very first call --
        // allow-all cold start, same shape as the flat path's min_samples guard).
    }
    return c.cutoff.load(std::memory_order_relaxed);
}
// Returns true = allow the promotion the caller already decided to make;
// false = veto it. is_attn_pilot selects which pilot's OWN per-layer
// counters to judge against -- DEPTH-1 and ATTN-PILOT predict the same
// target layer from different inputs and routinely have different
// per-layer accuracy, so one pilot's reliability at a layer says nothing
// about the other's. Falls back to "allow" (no veto) whenever the gate is
// off, the layer index is out of the tracked range, or fewer than
// min_samples live checks exist for this layer yet -- the same cold-start
// safety shape ram_vram_promote_adaptive's own min_samples floor uses:
// no real signal yet means don't act on it, not "assume the worst".
inline bool layer_promote_allowed(int layer_idx, bool is_attn_pilot) {
    const auto& cfg = layer_promote_gate_config();
    if (!cfg.enabled) return true;
    if (layer_idx < 0 || layer_idx >= PerfCounters::kMaxTrackedLayersForMiss) return true;
    auto& p = perf();
    uint32_t checks = is_attn_pilot
        ? p.n_pilot_attn_recall_checks_by_layer[layer_idx].load(std::memory_order_relaxed)
        : p.n_pilot_recall_checks_by_layer[layer_idx].load(std::memory_order_relaxed);
    if (checks < (uint32_t)cfg.min_samples) return true;
    uint32_t hits = is_attn_pilot
        ? p.n_pilot_attn_recall_hits_by_layer[layer_idx].load(std::memory_order_relaxed)
        : p.n_pilot_recall_hits_by_layer[layer_idx].load(std::memory_order_relaxed);
    uint32_t topk_sum = is_attn_pilot
        ? p.n_pilot_attn_recall_topk_sum_by_layer[layer_idx].load(std::memory_order_relaxed)
        : p.n_pilot_recall_topk_sum_by_layer[layer_idx].load(std::memory_order_relaxed);
    if (topk_sum == 0) return true;
    double acc = (double)hits / (double)topk_sum;
    if (cfg.top_fraction > 0.0 && cfg.top_fraction < 1.0) {
        // Rank-based "most confident layers only" mode -- see
        // rank_based_layer_accuracy_cutoff's own comment above.
        double cutoff = rank_based_layer_accuracy_cutoff(is_attn_pilot, cfg);
        return acc >= cutoff;
    }
    return acc >= cfg.threshold;
}

// predict_gpu1_rank_start/end + predict_gpu1_cap_per_layer (config): "Idea
// 1" prediction-based dual-role-slice loading (2026-08-16, see doc/gpu-
// utilization-vram-analysis.md). Deliberately a SEPARATE mechanism from
// RamVramPromoteConfig above, not a generalization of it -- that one
// targets the general/shared vault pool (rank 0/1 only, by design, per
// its own real-hardware tuning history: going past top_k=2 there was a
// measured regression from low-confidence promotions churning a shared,
// carefully-tuned cache). THIS one targets ranks 4+ specifically, but
// ONLY into the small, isolated dual-role slice (GpuTopology::
// dual_role_gpu_id()) via evict_lru's restrict_gpu_id -- a wrong bet here
// can only cost dual-role-slice churn, never touch the primary vault
// pool's own eviction candidates, which is why testing weaker ranks here
// is a defensible experiment even though the SAME weak-rank promotion
// idea already measured badly on the general pool.
struct Gpu1PredictPromoteConfig {
    bool enabled = false;
    int rank_start = -1;   // inclusive, 0-indexed (matches pilot_pred_ids[k])
    int rank_end = -1;     // inclusive
    int cap_per_layer = 1; // per LAYER TRANSITION, not per token -- see execution_layer.cpp's own local counter
};
inline Gpu1PredictPromoteConfig& gpu1_predict_promote_config() {
    static Gpu1PredictPromoteConfig v;
    return v;
}
inline void set_gpu1_predict_promote_config(bool enabled, int rank_start, int rank_end, int cap_per_layer) {
    auto& v = gpu1_predict_promote_config();
    v.enabled = enabled;
    v.rank_start = rank_start;
    v.rank_end = rank_end;
    v.cap_per_layer = cap_per_layer;
}

inline void dense_proj(StreamingEngine& engine, const std::string& wname,
                       const float* x, float* y, int S, int I, int O,
                       bool force_original_quant = false) {
    auto it = engine.resident_layers.find(wname);
    if (it == engine.resident_layers.end() || !it->second.host_ptr) {
        LOG_WARN("[DENSE] missing weight %s — zeroing output\n", wname.c_str());
        std::memset(y, 0, (size_t)S * O * sizeof(float));
    return;
  }
  auto& W = it->second;
  const bool ple_projection =
      wname == "blk.1.ple_key.weight" ||
      wname == "blk.1.ple_value.weight";
  const char* ple_diag_env =
      std::getenv("MOESTREAM_QWEN4EXP_PLE_PATH_DIAG");
  const bool ple_path_diag = ple_projection && ple_diag_env &&
                             ple_diag_env[0] == '1';
  const char* ple_cpu_env =
      std::getenv("MOESTREAM_QWEN4EXP_PLE_FORCE_CPU");
  const bool ple_force_cpu = ple_path_diag && ple_cpu_env &&
                             ple_cpu_env[0] == '1';
  if (ple_path_diag) {
      LOG_INFO("[QWEN4EXP-PLE-PATH] name=%s S=%d I=%d O=%d type=%u "
               "bytes=%zu gpu_fast=%d device=%d force_cpu=%d\n",
               wname.c_str(), S, I, O, static_cast<unsigned>(W.meta.type),
               W.meta.size_bytes, W.gpu_fast_ptr ? 1 : 0,
               W.device_ptr ? 1 : 0, ple_force_cpu ? 1 : 0);
  }
    // 1) GPU dispatch. Prefer the BF16-matched fast-path buffer
    //    (upload_dequantized_bf16, cuda_manager.hpp) when present — cuBLAS
    //    GEMV (matched BF16 weight/BF16 activation, F32 accumulate/output)
    //    instead of W's original quant type's fused per-block kernel, which
    //    has no GEMV specialization and is disproportionately slow at S=1
    //    decode (attn_output's Q5_K case, see CLAUDE.md). Falls through to
    //    the original device_ptr/meta.type path (fused kernel for
    //    K-quants/IQ, cuBLAS for F32/F16/BF16/Q8_0) when no fast-path
    //    buffer was built for this weight (not attempted, or didn't fit in
    //    the VRAM budget at load time).
    //
    //    Third attempt at this fast-path, per CLAUDE.md's full history: an
    //    F32 version worked but covered too few layers within budget; a
    //    BF16 version using MISMATCHED types (BF16 weight / F32 activation)
    //    caused real NaN corruption, was reverted, and was later confirmed
    //    (Test 9, main_test.cpp, real hardware) to be cleanly REJECTED by
    //    cublasGemmEx regardless — a dead end independent of the earlier
    //    corruption, which was separately root-caused to an unrelated
    //    concurrency race (now fixed) and never actually caused by BF16.
    //    This version uses the MATCHED-type combination (BF16 weight AND
    //    BF16 activation, dense_matmul_bf16_matched_host), verified correct
    //    in isolation via Test 10 (real hardware) before being wired in
    //    here.
    if (!ple_force_cpu && !force_original_quant && !cpu_only_mode() &&
        engine.gpu && W.gpu_fast_ptr &&
        engine.gpu->dense_matmul_bf16_matched_host(W.gpu_fast_ptr, x, y, S, I, O)) {
        if (ple_path_diag)
            LOG_INFO("[QWEN4EXP-PLE-PATH] selected=BF16-fast\n");
        return;
    }
    if (!ple_force_cpu && !cpu_only_mode() && engine.gpu && W.device_ptr &&
        engine.gpu->dense_matmul_host(W.meta.type, W.device_ptr, x, y, S, I, O)) {
        if (ple_path_diag)
            LOG_INFO("[QWEN4EXP-PLE-PATH] selected=original-weight-GPU\n");
        return;
    }
    if (ple_path_diag)
        LOG_INFO("[QWEN4EXP-PLE-PATH] selected=CPU-reference\n");
    // 2) CPU reference dequant-matmul: correct for F32/F16/BF16/Q8_0/Q4_0;
    //    loudly zeroes K-quants/IQ2 (dequant bodies pending). Never the old
    //    silent "reinterpret quant as float". This is routine/expected for
    //    many quant types today (no GPU kernel yet) and fires once per weight
    //    per forward pass — DEBUG level, not a real problem on its own.
    LOG_DEBUG("[DENSE] %s type=%u: no GPU dispatch — CPU dequant-matmul fallback\n",
              wname.c_str(), (unsigned)W.meta.type);
    if (!cpu_quant_matmul(W.meta.type, W.host_ptr, x, y, S, I, O)) {
        LOG_ERROR("[DENSE] %s type=%u: CPU dequant unavailable — some outputs zeroed\n",
                  wname.c_str(), (unsigned)W.meta.type);
    }
}

// Narrow Qwen4Exp shared-expert bridge.  The normal dense_proj() contract is
// intentionally unchanged: every call owns its activation upload.  These
// helpers are used only by run_qwen4exp_shared_host() for the three
// projections which read the same normed activation.  A false return means
// "use dense_proj normally" (BF16 fast path, missing device tensor, or a
// dispatch failure), never "the projection was completed".
inline bool dense_upload_shared_input(StreamingEngine& engine,
                                      const float* x, int S, int I) {
    if (cpu_only_mode() || !engine.gpu || !x) return false;
    return engine.gpu->dense_matmul_host_upload_input(x, S, I);
}

inline bool dense_proj_from_shared_input(StreamingEngine& engine,
                                         const std::string& wname,
                                         float* y, int S, int I, int O) {
    auto it = engine.resident_layers.find(wname);
    if (it == engine.resident_layers.end() || !it->second.host_ptr ||
        cpu_only_mode() || !engine.gpu || !it->second.device_ptr ||
        it->second.gpu_fast_ptr)
        return false;
    return engine.gpu->dense_matmul_host_reuse_input(
        it->second.meta.type, it->second.device_ptr, y, S, I, O);
}

// F32 RMSNorm via GPU (norm weights are offloaded to VRAM), CPU rmsnorm fallback.
inline void rmsnorm_proj(StreamingEngine& engine, const std::string& wname,
                         const float* x, float* y, int S, int D, float eps) {
    auto it = engine.resident_layers.find(wname);
    if (it == engine.resident_layers.end() || !it->second.host_ptr) {
        LOG_WARN("[NORM] missing weight %s — passing input through\n", wname.c_str());
        std::memcpy(y, x, (size_t)S * D * sizeof(float));
        return;
    }
    auto& W = it->second;
    if (!cpu_only_mode() && engine.gpu && W.device_ptr &&
        engine.gpu->rmsnorm_gpu_host(W.device_ptr, x, y, S, D, eps)) {
        return;
    }
    for (int s = 0; s < S; ++s)
        rmsnorm_row(y + (size_t)s * D, x + (size_t)s * D,
                    reinterpret_cast<float*>(W.host_ptr), D, eps);
}
