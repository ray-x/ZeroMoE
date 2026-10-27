# Timeline of an MTP verify window (Qwen4Exp, promotion on)

Capture: `mtp_nsys.sh` (nsys, `--trace=cuda,nvtx,osrt --cuda-graph-trace=node`), palindrome prompt, MTP with 3
drafts, `qwen4exp_expert_promotion=on`, 3090 only; 33 steady windows. NVTX ranges: `mtp.verify_window`,
`win.run/submit/sync/readback/commit`, `promo.step`, `mtp.draft_step`, `mtp.sampling` (`core/nvtx_range.hpp`).
Analyzer: `tools/nsys_window_breakdown.py <export.sqlite>`. The raw trace is in
`~/Projects/ZeroMoE-results/promotion-study/nsys/`. nsys stretches the window from 41.9 ms (unprofiled) to
55.7 ms; multiply by ~0.75 to compare with the unprofiled profile.

## One window, median, nsys time (sums to the wall)

| part | ms | share |
|---|---|---|
| GPU kernels running | 19.7 | 35% |
| GPU copies only (no kernel running) | 3.7 | 7% |
| host waiting in sync/query calls, GPU idle | 1.9 | 3% |
| host in other CUDA calls, GPU idle (graph launches, async copies) | 14.6 | 26% |
| host only: no CUDA call, GPU idle (CPU expert compute, scheduler) | 17.3 | 31% |
| **wall** | **55.7** | |

## What the window consists of

* ~4,620 GPU kernels per window; 38% of the kernel time is in kernels shorter than 8 us. The biggest groups:
  quantized mat-vec 4.4 ms (302 launches), bf16 mat-mul 4.0 ms (243), MoE mat-vec 3.2 ms (96), elementwise
  `k_bin_bcast` 2.3 ms in **1,532 launches**, GDN 0.75 ms (144), quantize 0.55 ms (398), scale 0.45 ms (387).
* The scheduler splits the window into **49 CUDA graphs** (about one per MoE layer: GPU part, CPU cold-expert
  part, GPU part). Each is launched separately: 49 `cudaGraphLaunch` calls cost **12.7 ms** (about 0.26 ms
  each, roughly 2.7 us per node), plus 344 async copies (2.1 ms) and 441 `cudaStreamSynchronize` calls.
* Waiting in synchronization calls while the GPU is idle is small (1.9 ms): the host spends its idle time
  launching graphs and computing cold experts, not polling.

## Correction: most of the launch cost is the profiler

`--cuda-graph-trace=node` makes every `cudaGraphLaunch` cost 174-630 us (median 246) because nsys tracks each
node; plain decode and the window have a similar node count (about 5,000 compute nodes; see
`ZEROMOE_GRAPH_OP_HISTOGRAM`). The unprofiled window is 41.9 ms against 55.7 ms under nsys, a difference of
13.8 ms that matches the 14.6 ms attributed to "other CUDA calls". Without the artifact the window is about
GPU kernels 19.7 + copies 3.7 + sync waits 1.9 + host-only 17 ms. Launch overhead is therefore a small lever;
the 17 ms of host-only time (cold-expert compute and scheduling, GPU idle) is the largest non-GPU part. With the
early-ready lane on in windows the picture barely changes (window 52.5 ms under nsys, host-only 15.9 ms).

## Reading (superseded in part by the correction above)

Only a third of the window is GPU compute. The rest is per-layer overhead: launching ~4,600 small kernels
through 49 graph launches, and the serialized CPU split for the cold experts. Faster GPU kernels would not move
this; fewer launches/splits and fewer cold experts would.

## Host-side CPU profile (perf, palindrome, MTP + promotion, 3090, 10 OpenMP threads)

`perf record -F 1999` over the decode phase (about 7.3 s, 138k samples, `mtp_prom_run.sh` setup; sampling delayed
until just before decode). By thread:

| thread | share of its samples |
|---|---|
| 9 OpenMP workers (each ~91% of the decode window) | 67% libgomp barrier/spin wait, 33% ggml kernels (essentially all `ggml_vec_dot_q2_0_q8_0`) |
| main thread | 53% CUDA driver (graph launches, sync polling), 25% ggml kernels, 19% other, 3% spin |

Reading: the Q2_0 cold-expert compute is real work, about 30% of wall time on every worker, which matches the
~11-12 ms per window the hot-set sweeps attributed to cold experts. The rest of the CPU time is waiting: workers
spin at a barrier for every small op of the cold branch (about 2-3 cold experts per layer and window) while the
main thread drives the GPU. The Q2_0 dot already has a ZeroMoE AVX-512 VNNI path (1-row and 4-row variants,
runtime-dispatched on `avx512vnni`); a faster dot is not the lever. Thread count is not either (6 / 8 / 10 threads
gave 59.8 / 58.0 / 58.5 tok/s on palindrome and 57.5 / 55.6 / 57.4 on math, inside the noise).

Levers this leaves on the CPU side: fewer cold evaluations (promotion; the hot-set size is capped near 13,000 by
the draft and window memory: K=13,500 and 14,000 fail draft admission), and fewer barriers per layer by fusing the
cold branch's ops (gate/up/down matmuls, SwiGLU, activation quantization).
