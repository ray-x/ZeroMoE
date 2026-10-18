// hyper_connections.hpp — pure-math primitives for DeepSeek-V4's
// Hyper-Connections (multi-stream residual), extracted from
// src/hyper_connections.cpp's engine-facing hc_pre_collapse/hc_post_combine
// so the Sinkhorn normalization specifically is unit-testable in isolation
// (tests/test_components.cpp), mirroring rope.hpp's own "primitives here,
// engine glue in the .cpp" split. Dependency-free beyond <cmath>/<algorithm>
// -- no engine/CUDA/AVX-512 headers -- so it compiles in the plain CPU test
// harness on any platform.
//
// UPDATE 2026-08-08: held directly side-by-side against llama.cpp's real
// deepseek4.cpp (build_hc_pre/build_hc_sinkhorn/build_hc_post/build_hc_head,
// commit 1269cb1 as of that checkout) after a real-hardware run confirmed a
// separate multi-GPU __constant__-memory bug's fix did NOT resolve
// DeepSeek-V4-Flash's garbled output -- meaning a second, independent bug
// was still active, and this file's own "more tentative than most of this
// project's unverified work" self-assessment made it the top suspect. Found
// and fixed two real discrepancies from that comparison (see
// sinkhorn_normalize_comb's own comment for the Sinkhorn one); the mixing-
// coefficient layout (pre-gate/post-gate/comb order in mix_dim) and the
// pre-collapse/post-combine formulas below WERE confirmed correct as
// originally reconstructed. STILL UNVERIFIED ON REAL HARDWARE (no rebuild/
// rerun yet since this fix) -- source-comparison-confirmed, not yet
// hardware-confirmed.
//
// Reference (per the research this was built from): DeepSeek-V4's
// "Hyper-Connections" (https://arxiv.org/pdf/2512.24880) replace the
// single-stream residual add with hc_mult (e.g. 4) parallel streams. Per
// sub-block (attention and FFN each get their own call):
//   1. RMS-norm the flattened [hc_mult*D] multi-stream vector (per token).
//   2. ONE matmul (hc_{attn,ffn}_fn weight, shape {hc_mult*D, hc_mix_dim}
//      where hc_mix_dim = (2+hc_mult)*hc_mult) + a learned per-coordinate
//      affine (hc_{attn,ffn}_scale, hc_{attn,ffn}_base -- CONFIRMED present
//      in the real tensor list, applied to each of the 3 groups below
//      BEFORE their activation) produces ALL mixing coefficients in one
//      shot: pre-gate (hc_mult values), post-gate (hc_mult values), comb
//      matrix (hc_mult*hc_mult values), laid out in that order.
//   3. pre-gate -> affine -> sigmoid -> +eps (CONFIRMED: real code adds eps
//      AFTER sigmoid here, via ggml_scale_bias); used to collapse the
//      hc_mult streams into ONE input vector for the sub-block via a
//      weighted sum.
//   4. post-gate -> affine -> sigmoid*2 (no +eps here, confirmed -- only
//      pre-gate and the final head/output-collapse gate get it); used after
//      the sub-block runs, to scale how much of ITS output writes back into
//      each stream.
//   5. comb -> affine -> Sinkhorn-normalized (this header's
//      sinkhorn_normalize_comb): softmax over the dst axis + eps, then ONE
//      unconditional row-normalize, then hc_sinkhorn_iters-1 rounds of
//      column-then-row normalization (see that function's own comment --
//      this exact structure was the discrepancy found+fixed 2026-08-08).
//   6. new_stream[dst] = sub_block_out * post_gate[dst]
//                       + sum_src comb[dst][src] * old_stream[src]
#pragma once
#include <cmath>
#include <algorithm>

inline float hc_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Sinkhorn-normalizes an hc x hc matrix IN PLACE, row-major
// (comb[dst*hc + src]).
//
// CONFIRMED against llama.cpp's real deepseek4.cpp (build_hc_sinkhorn),
// 2026-08-08 -- this superseded an earlier reconstruction that ran `iters`
// full column+row rounds with no unconditional leading step; the real
// sequence is: softmax+eps, then exactly ONE unconditional "row" round
// (their norm_cols(), which sums over src for each dst -- our naming calls
// this normalize_rows, see below), THEN `iters-1` rounds of column-then-row.
// With small iters (2-3, typical) the old version produced a measurably
// different, non-converged doubly-stochastic matrix -- and this matrix
// directly weights the residual-stream mixing on every single layer, so the
// error compounded layer-over-layer. This is suspected as a real
// contributor to the still-unresolved DeepSeek-V4-Flash garbled-output bug.
//
// Step 1: for each FIXED src (column), softmax the hc dst-values in that
// column (matches ggml_soft_max operating over the dst axis), then add a
// small additive eps to every entry (matches deepseek4.cpp's
// `ggml_add(comb, eps)` immediately after the softmax).
//
// Step 2 naming note: llama.cpp's own "norm_cols"/"norm_rows" names refer to
// ggml's [dst,src,nt] axis layout and read backwards relative to the
// row-major dst*hc+src indexing used here -- to avoid propagating that
// confusion, this file names each helper by WHAT it makes true instead:
// normalize_rows() makes each dst's row (fixed dst, varying src) sum to 1;
// normalize_columns() makes each src's column (fixed src, varying dst) sum
// to 1. The real call sequence, translated: normalize_rows() once
// (unconditional, runs even if iters==0 or 1), then `iters-1` rounds of
// (normalize_columns(), normalize_rows()).
//
// Testable property (tests/test_components.cpp): after Step 2, every row
// AND every column sums to ~1 (the defining property of a doubly-stochastic
// matrix) -- pure math, no engine dependency, checkable with a random
// matrix and no reference implementation needed.
inline void sinkhorn_normalize_comb(float* comb, int hc, int iters, float eps) {
    if (hc <= 0) return;

    // Step 1: softmax over dst (rows) independently for each src (column).
    for (int src = 0; src < hc; ++src) {
        float max_v = comb[0 * hc + src];
        for (int dst = 1; dst < hc; ++dst)
            max_v = std::max(max_v, comb[dst * hc + src]);
        float sum = 0.0f;
        for (int dst = 0; dst < hc; ++dst) {
            float e = std::exp(comb[dst * hc + src] - max_v);
            comb[dst * hc + src] = e;
            sum += e;
        }
        float inv = 1.0f / std::max(1e-12f, sum);
        for (int dst = 0; dst < hc; ++dst)
            comb[dst * hc + src] = comb[dst * hc + src] * inv + eps;
    }

    // Step 2: one unconditional row-normalize, then (iters-1) rounds of
    // column-then-row -- matches deepseek4.cpp's build_hc_sinkhorn exactly
    // (its norm_cols() once, then `for i=1;i<iters` of norm_rows()+norm_cols()).
    auto normalize_rows = [&]() {
        for (int dst = 0; dst < hc; ++dst) {
            float sum = 0.0f;
            for (int src = 0; src < hc; ++src) sum += comb[dst * hc + src];
            float inv = 1.0f / (sum + eps);
            for (int src = 0; src < hc; ++src) comb[dst * hc + src] *= inv;
        }
    };
    auto normalize_columns = [&]() {
        for (int src = 0; src < hc; ++src) {
            float sum = 0.0f;
            for (int dst = 0; dst < hc; ++dst) sum += comb[dst * hc + src];
            float inv = 1.0f / (sum + eps);
            for (int dst = 0; dst < hc; ++dst) comb[dst * hc + src] *= inv;
        }
    };

    normalize_rows();
    for (int it = 1; it < iters; ++it) {
        normalize_columns();
        normalize_rows();
    }
}
