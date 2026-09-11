// moe_decode_batch.hpp — prototype: grouped, low-overhead decode-time (S==1)
// routed-expert FFN dispatch.
//
// CONTEXT (2026-09-10 decode-gap investigation): MoEStream's real decode
// bottleneck is NOT expert compute -- a live measurement on qwen4exp found
// the `expert` bucket at ~77% of decode wall time while the actual GPU
// matmul time inside it (`gpu=[up/cmp/dn]`) was only ~7% of that bucket.
// The rest is per-expert dispatch orchestration: execution_layer.cpp's
// `dispatch_expert_k` lambda re-derives a fresh cache-residency lookup
// (`find_expert`), a lock acquisition (`expert_ffn_resolve_gpu_loc` ->
// `hold_for_compute`), and promotion/pin bookkeeping (`ram_pin`) for EVERY
// ONE of topk (10) x n_layers (48) = 480 expert lookups, every decode
// token -- even when the whole expert set is already 100% resident
// (host_ram_gb + ram_preload_all_experts => no possible miss for the
// session's lifetime). beellama/llama.cpp instead decides weight placement
// ONCE at load time (`-ncmoe`) and runs one static ggml_mul_mat_id graph
// per layer with zero per-token placement decisions
// (llm_graph_context::build_moe_ffn, llama-graph.cpp) -- see this
// project's own [[llamacpp-op-offload-is-the-lever]] /
// [[moe-expert-dispatch-not-batched]] memory notes.
//
// This header prototypes the computational core of a decode-time fast path
// that collapses to the SAME math as today's per-expert loop
// (execution_layer.cpp's `expert_ffn_cpu`, called once per k inside
// `dispatch_expert_k`) but takes ALREADY-RESOLVED weight pointers and a
// pair of REUSED scratch buffers, so it can run for a whole token's topk
// experts with none of the per-expert lock/residency/allocation overhead
// above. It is deliberately F32-only and CUDA-free (mirrors this project's
// own "prove the idea with a portable CPU reference first" convention,
// e.g. qwen4exp_gdn_delta_chunk vs qwen4exp_gdn_delta_step) -- production
// integration still needs the real quantized-matmul dispatch
// (cpu_quant_matmul / gpu_matmul_dispatch) and a real "is this layer's
// whole expert set actually resident" precondition check, neither of which
// this prototype attempts. See the accompanying unit test
// (tests/test_components.cpp, "moe_decode_batch grouped vs per-expert
// loop") for the correctness argument this is meant to support, and this
// investigation's own chat history for the NOT-YET-DONE integration plan
// (gating this on a per-layer "all topk resident" fast check inside
// moe_forward_gguf, execution_layer.cpp).
//
// NOT wired into the live dispatch path. Standalone and additive so it can
// be reviewed/tested/discarded independently of execution_layer.cpp, which
// another session is concurrently modifying.
#pragma once
#include "kernels_cpu.hpp" // matmul, silu_inplace, elem_mul (already linked by every target that needs this file)

// Computes output[D] = sum_k expert_weight[k] * SwiGLU_FFN_k(hidden), for
// k_used already-selected, already-weighted routed experts, in one call.
//
// Weight layout matches the production convention (kernels_cpu.hpp's
// matmul(): W is [O,I] row-major): expert_gate[k]/expert_up[k] are
// [moe_inter, D], expert_down[k] is [D, moe_inter].
//
// `output` is accumulated into (NOT zeroed here) so callers can add a
// shared-expert or residual contribution before/after calling this,
// exactly like the existing per-expert loop's own
// `out_row[d] += expert_out[d] * w` convention.
//
// `gate_scratch`/`up_scratch` (each >= moe_inter floats) and `down_scratch`
// (>= D floats) are caller-owned and REUSED across all k_used experts --
// this is the concrete "no per-expert allocation" half of the idea; the
// "no per-expert lock/residency lookup" half is a property of the CALLER
// (it must have already established that expert_gate[k]/up[k]/down[k] are
// valid, resident pointers before calling this -- this function performs
// no residency checks of its own, by design).
//
// swiglu_clamp == 0.0f is treated as "no clamp", matching this project's
// existing ArchitectureConfig::swiglu_clamp_exp convention (empty/0 = off)
// -- clamping itself is deliberately NOT implemented here since neither
// qwen4exp nor this prototype's own unit test need it; a real integration
// into moe_forward_gguf would need to add it (mirroring
// swiglu_clamp_inplace) before this could be enabled for a clamped model.
inline void moe_decode_expert_ffn_batched(
    const float *hidden, float *output, const float *const *expert_gate,
    const float *const *expert_up, const float *const *expert_down,
    const float *expert_weight, int k_used, int D, int moe_inter,
    float *gate_scratch, float *up_scratch, float *down_scratch) {
  for (int k = 0; k < k_used; ++k) {
    matmul(gate_scratch, hidden, expert_gate[k], 1, D, moe_inter);
    matmul(up_scratch, hidden, expert_up[k], 1, D, moe_inter);
    silu_inplace(gate_scratch, moe_inter);
    elem_mul(gate_scratch, gate_scratch, up_scratch, moe_inter);
    matmul(down_scratch, gate_scratch, expert_down[k], 1, moe_inter, D);
    const float w = expert_weight[k];
    for (int d = 0; d < D; ++d)
      output[d] += down_scratch[d] * w;
  }
}
