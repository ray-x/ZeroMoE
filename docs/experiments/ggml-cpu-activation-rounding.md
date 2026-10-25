# Experiment: CPU vs CUDA activation rounding in ggml (Ling placement drift)

Status: **research only. Nothing here is applied to the ZeroMoE ggml series or any build.** The patches live in
`patches/experimental/` (which no build script reads) and are kept for reference and for a possible upstream report.

Decision (2026-10-23): a mismatch of about 0.96-0.99 cosine on Ling between placements that move experts between CPU and
GPU is accepted. Ling's phase-cache gate is `scripts/check_kernel_numerics.sh` plus a bitwise-identical repeated request
(`bench/phase_cache_transition.py`). MiMo keeps the 0.999 floor. The kernel-level fix was rejected on performance (below).

## What was observed

Moving experts between the CPU and the GPU (the phase-cache extension, an enlarged permanent GPU cache, or llama.cpp with
a few layers offloaded) changes Ling's logits: cosine 0.95-0.99 against the other placement, top-1 unchanged. Stock
llama.cpp shows the same thing (CPU vs CPU+10 GPU layers: .952 .969 .978 .987 .994 .994 .994 .996 over 8 frames), so it is
not a ZeroMoE bug.

## Cause (measured)

* The kernels are exact. With activations that 8-bit rounding represents exactly, CPU and CUDA match float64 to <=4e-5
  for IQ1_S, IQ2_XXS, IQ2_S, Q2_K and Q3_K (`scripts/check_kernel_numerics.sh`, `EXACT_INPUTS=1`). Aliased expert views at
  the engine's slot strides are bitwise identical to contiguous tensors.
* Every quantized matmul first rounds the activation vector to 8 bits. CPU uses `Q8_K` (one scale per 256 values), CUDA
  `Q8_1` (one per 32). On Gaussian activations both give about 0.7% error; on Ling's swiglu output, which is spiky (block
  max / block rms = 7.3 against about 3 for a Gaussian), the 256-value scale is set by an outlier and the CPU error doubles.
* Real weights (layer 63, `tests/test_real_tensor_numerics.cpp`), error against float64 on the same quantized weights:

  | op | CPU | CUDA | CPU vs CUDA |
  |---|---|---|---|
  | gate IQ1_S | 0.93% | 0.81% | 1.16% |
  | up IQ1_S | 0.99% | 0.87% | 1.23% |
  | swiglu | 1.31% | 1.23% | 1.70% |
  | down IQ2_XXS, same exact input | **1.71%** | **0.94%** | 1.76% |

  Exact float64 math on the swiglu output rounded to 256-value blocks gives 1.71%, rounded to 32-value blocks 0.94%:
  identical to the measured kernels. The gap is the down-projection's input rounding; swiglu itself adds no error.
* llama.cpp's node-by-node diff (`LLAMA_NODE_DUMP`, `tools/node_dump_diff.cpp`) shows the same: identical until the first
  GPU layer, dense matmuls 0.2-0.4%, routed-expert gate/up 0.9%, swiglu 1.4%, down 2.3%.

## Prototype fix and result

`patches/experimental/ggml-cpu-act32-prototype.diff` (llama.cpp `8baabd53b`): IQ1_S, IQ2_XXS and IQ2_S dot products read a
`Q8_0` activation (32-value blocks) with exactly dequantized weights, and the tiled CPU kernel (which assumes `Q8_K`) is
excluded for those types. `patches/experimental/ggml-cpu-act32-down-only.diff` does it for IQ2_XXS only (Ling's down), still with the slow dequantizing dot product.
`patches/experimental/ggml-cpu-act32-down-only-avx2.diff` is the same IQ2_XXS-only change with a vectorized AVX2 `ggml_vec_dot_iq2_xxs_q8_0` (this is the kernel that was benchmarked below).

CPU run vs CPU+10-GPU-layer run, same build, 8 teacher-forced frames:

| build | cosine per frame |
|---|---|
| stock | .9522 .9689 .9775 .9865 .9943 .9939 .9944 .9956 |
| all three types, 32-value blocks | .9999 .9998 .9999 1.0 1.0 1.0 1.0 1.0 |
| down-projection only | .9999 .9999 .9999 .9999 1.0 .9999 .9999 1.0 |

The whole drift is the down-projection input. Caveats: the prototype dequantizes each weight row per dot product (slow), and
only Ling was measured.

## Why it is not applied

The vectorized IQ2_XXS x Q8_0 AVX2 kernel in `ggml-cpu-act32-down-only-avx2.diff` (same integer work; one int->float convert and FMA per 32-value sub-block) gives the
same numbers as the prototype but costs **+8% (10 threads: 739 -> 800 us) and +6-7% (1 thread: 3.80 -> 4.06 ms)** on a
decode-shaped op (K=1280, N=4096, 8 of 64 experts, 1 token). Performance of the CPU expert path matters more to ZeroMoE than
placement-invariant numerics, so it was rejected. End-to-end Ling decode with the kernel was not timed.

## Suggestions, not done

1. Finer rounding only where it matters: the gap is in the down input; gate/up were already close. An adaptive scheme could
   use 32-value blocks only for blocks whose max/rms ratio is high.
2. A `Q8_K` variant with 32-value scales that keeps the block sums, so the existing vectorized IQ kernels need small edits
   (the clean route to a mergeable kernel; real engineering work).
3. fp32 activations with on-the-fly weight dequantization for the 1-2-bit experts: most accurate, more arithmetic; may be
   hidden if the offload path is memory-bound (untested).
4. Make CUDA round like the CPU (placement-invariant, but both less accurate): not an ggml improvement.
5. Better scale selection on `Q8_K` (clipping outliers): expected to help less than finer blocks.
6. Before any upstream PR: perplexity/KL against a high-precision reference on Ling and one more IQ model (so far only CPU/GPU
   agreement is shown, not model quality), a speed comparison on real decode, and a way to keep the tiled kernel. An upstream
   issue with this data is the cheaper first step.
7. Unexplained: CUDA's IQ1_S gate/up error on layer 7 is 2.3% against 1.2% for the 32-block CPU, so CUDA is not exactly
   "32-value blocks" there (possibly the mmq path's block layout).

## Reproduce

* Kernel check: `scripts/check_kernel_numerics.sh [build-dir]`.
* Real-weight test: build `real_tensor_numerics_test` (or link `tests/test_real_tensor_numerics.cpp` against a ggml build) and run
  `real_tensor_numerics_test <ling-model-dir> <node dump from a CPU run> [layer]`.
* llama.cpp logits: `tools/ling_llama_logits.cpp` (`LLAMA_FORCE_TOKENS` teacher forcing, `LLAMA_NGL` GPU layers,
  `LLAMA_NODE_DUMP` per-node dump); diff dumps with `tools/node_dump_diff.cpp`.
* Prototype build: `git worktree add ../llama.cpp-act32 8baabd53b`, `patch -p1 < patches/experimental/ggml-cpu-act32-prototype.diff`
  (the diffs are relative to the llama.cpp repo root), build with CUDA for the GPU half. Use `ggml-cpu-act32-down-only-avx2.diff`
  for the benchmarked kernel.

Numbers are from the session notes in `docs/handoff-2026-10-23-phase-cache.md`.
