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

## Reading

Only a third of the window is GPU compute. The rest is per-layer overhead: launching ~4,600 small kernels
through 49 graph launches, and the serialized CPU split for the cold experts. Faster GPU kernels would not move
this; fewer launches/splits and fewer cold experts would.
