// hyper_connections.hpp — pure-math primitives for DeepSeek-V4's
// Hyper-Connections (multi-stream residual), extracted from
// src/hyper_connections.cpp's engine-facing hc_pre_collapse/hc_post_combine
// so the Sinkhorn normalization specifically is unit-testable in isolation
// (tests/test_components.cpp), mirroring rope.hpp's own "primitives here,
// engine glue in the .cpp" split. Dependency-free beyond <cmath>/<algorithm>
// -- no engine/CUDA/AVX-512 headers -- so it compiles in the plain CPU test
// harness on any platform.
//
// UNVERIFIED ON REAL HARDWARE, and more tentative than most of this
// project's "unverified" work: the exact Sinkhorn procedure and mixing-
// coefficient layout below are a best-effort reconstruction from reading
// llama.cpp's deepseek4.cpp (build_hc_pre/build_hc_sinkhorn/build_hc_post),
// not a byte-exact transcription with that source held side-by-side during
// implementation. Treat every formula/layout choice here as a hypothesis to
// confirm against real DeepSeek-V4 output, not an established fact the way
// e.g. rope.hpp's interleaved-vs-NeoX convention already was (that one WAS
// confirmed against llama.cpp source directly, twice, per its own comment).
//
// Reference (per the research this was built from): DeepSeek-V4's
// "Hyper-Connections" (https://arxiv.org/pdf/2512.24880) replace the
// single-stream residual add with hc_mult (e.g. 4) parallel streams. Per
// sub-block (attention and FFN each get their own call):
//   1. RMS-norm the flattened [hc_mult*D] multi-stream vector (per token).
//   2. ONE matmul (hc_{attn,ffn}_fn weight, shape {hc_mult*D, hc_mix_dim}
//      where hc_mix_dim = (2+hc_mult)*hc_mult) + bias (hc_{attn,ffn}_base)
//      produces ALL mixing coefficients in one shot: pre-gate (hc_mult
//      values), post-gate (hc_mult values), comb matrix (hc_mult*hc_mult
//      values), laid out in that order in the mix_dim output.
//   3. pre-gate -> sigmoid; used to collapse the hc_mult streams into ONE
//      input vector for the sub-block via a weighted sum.
//   4. post-gate -> sigmoid*2; used after the sub-block runs, to scale how
//      much of ITS output writes back into each stream.
//   5. comb -> Sinkhorn-normalized (this header's sinkhorn_normalize_comb):
//      initial softmax over the dst axis + eps, then hc_sinkhorn_iters
//      rounds of alternating column-then-row normalization.
//   6. new_stream[dst] = sub_block_out * post_gate[dst]
//                       + sum_src comb[dst][src] * old_stream[src]
#pragma once
#include <cmath>
#include <algorithm>

inline float hc_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Sinkhorn-normalizes an hc x hc matrix IN PLACE, row-major
// (comb[dst*hc + src]).
//
// Step 1 (interpretation of "initial softmax over dst, +eps"): for each
// FIXED src (column), softmax the hc dst-values in that column, then add a
// small additive eps to every entry (smoothing so no entry is ever exactly
// zero going into the iterative normalization below -- avoids a 0/0 on a
// degenerate all-zero row/column during Step 2).
//
// Step 2: `iters` rounds of alternating column-normalize (each column of hc
// dst-values divided by its own sum, so every column sums to 1) then row-
// normalize (each row of hc src-values divided by its own sum, so every row
// sums to 1) -- the standard Sinkhorn-Knopp iteration toward a doubly-
// stochastic matrix. iters==0 leaves the matrix as the raw (softmax+eps)
// result from Step 1, not further normalized -- matches
// ArchitectureConfig::hc_sinkhorn_iters defaulting to 0 (feature off) for
// every non-DeepSeek-V4 file, though this whole function is only ever
// called when hc_mult>1 in the first place.
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

    // Step 2: alternating column/row normalization.
    for (int it = 0; it < iters; ++it) {
        // Normalize columns: for each src, sum over dst == 1.
        for (int src = 0; src < hc; ++src) {
            float sum = 0.0f;
            for (int dst = 0; dst < hc; ++dst) sum += comb[dst * hc + src];
            float inv = 1.0f / std::max(1e-12f, sum);
            for (int dst = 0; dst < hc; ++dst) comb[dst * hc + src] *= inv;
        }
        // Normalize rows: for each dst, sum over src == 1.
        for (int dst = 0; dst < hc; ++dst) {
            float sum = 0.0f;
            for (int src = 0; src < hc; ++src) sum += comb[dst * hc + src];
            float inv = 1.0f / std::max(1e-12f, sum);
            for (int src = 0; src < hc; ++src) comb[dst * hc + src] *= inv;
        }
    }
}
