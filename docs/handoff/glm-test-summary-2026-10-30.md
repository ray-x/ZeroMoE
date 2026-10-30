# GLM-5.3-Flash GSQ-RCO: test summary (2026-10-30)

Everything below was measured on this box, which other agents share (builds and model runs overlap); every speed figure carries its contamination report
(`bench/glm_gsqrco_pool.sh` prints which other processes used >= 20% CPU during the run) and a run is only quoted as valid when no compile tools were active.
Goal that was set: GLM decode toward ~20 tok/s, 3090 first, then 3090 + 5060 Ti. **Result: 14.1-14.5 tok/s on the 3090 + 5060 Ti; 20 tok/s not reached.**

## Setup

| | |
|---|---|
| Hardware | Xeon W-2235 (6C/12T, AVX-512 + VNNI, ~60 GB/s DRAM read), 125 GB RAM, RTX 3090 (24 GB, PCIe 3.0 x16) + RTX 5060 Ti (16 GB, x8); no GPU peer-to-peer (`nvidia-smi topo -p2p r`: NS) |
| Model | `GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf`, 117.48 GB (109.41 GiB), **3.000 bpw**, Q2_K 83.3% / Q3_K 6.9% / Q4_K 6.9% of 313.327 B parameters. AJ IQ2_XXS (87.35 GB, 2.230 bpw) for one comparison |
| Config | offload-pool mode, `graph_offload_ddr_gib = 88`, 14.2 GiB hot experts on the 5060 Ti, 12.4 GB expert cache on the 3090, 10 CPU threads (prefill min(that, 6)), no GPU assist |
| Workload | 6 pilot prompts (`bench/data_glm_gsqrco_pilot6_prompts.txt`), 96 tokens each, 550 steady decode tokens (tokens 21+ of the engine's per-token time); teacher-forced numerics test: "The capital of France is" + 8 llama.cpp tokens |
| Code | master `48ddafa` (ggml llama.cpp `8baabd53b` + patches 0001-0017 + host-node 0001-0003); earlier rows name the build they ran on |

## 1. Decode speed

| Test | Result | Verdict |
|---|---|---|
| Shipped settings, clean runs | **69.0 / 69.4 / 69.7 / 70.2 / 70.5 / 70.8 ms = 14.1-14.5 tok/s** (best 68.9 ms = 14.51) | baseline |
| Decode threads 6 -> 10 | 76.9 -> 70.5 ms (13.0 -> 14.2 tok/s) | **kept** (default 10) |
| Pool 80 -> 88 GiB, 3090 only | 190 ms (5.3 tok/s, ~2 SSD misses/token) -> 93.8 ms (10.7 tok/s) | **kept**; the 10.7 was measured with a since-removed memory trick, **not re-measured** |
| AJ IQ2_XXS, same binary and prompts | 75.4 ms = 13.26 tok/s (GSQ-RCO 70.8) | the old 70.1 ms AJ record is not reproduced on current code |
| Ranking matched to the pilot prompts | 63.1 ms = 15.84 tok/s | **in-sample only** (ranking built from the same 6 prompts): an upper bound, not a general gain |
| K-quant GPU assist (1 slot) | 90.7 ms vs 76.9 | worse, off |
| More resident experts (5060 Ti hot 15.0 GiB; + 3090 cache 13.4 GB) | 70.1 and 68.9 vs 69.7 ms | no gain (noise) |
| Q2_K AVX512-VNNI dot, controlled A/B | patched 70.9 / 71.2 vs stock 69.0 / 69.4 ms | no gain, 1.9 ms slower; bit-exact and 1.11x in the microbenchmark but not in the engine; not adopted |
| 5060 Ti branch before the CPU lane marker (`4eb6825`), A/B | **B 69.8 ms (clean) vs A 70.5 ms (clean)**; A 72.4 (clangd load, marginal); a second B run 619.7 ms invalid (2,337 `cc1plus` samples) | numerically exact, **no measurable gain (-0.7 ms, noise)**; not shipped; see handoff |
| 3090 only, pool 84 | 9.42 and 8.11 tok/s, both under heavy compile load (63 and 711 contaminated samples) | **invalid**; the memory reading is valid: min 19.6-20.7 GiB MemAvailable |
| After the registration fix / after the planner margin | 70.8 ms / 70.2 ms | no regression |

## 2. Where one token goes (Nsight node trace, 98 tokens; the profiler stretches 70.8 to 87.9 ms, scale by ~0.8)

| Phase | ms | Share | Busy |
|---|---|---|---|
| CPU waits for the 3090 (attention, router) | 17.2 | 20% | 3090 86%, 5060 Ti idle |
| CPU cold experts + upload | 47.2 | 54% | 3090 6.7%, 5060 Ti 0%: both GPUs idle |
| GPU combine + 5060 Ti hot experts + token tail | 23.7 | 27% | 3090 18%, 5060 Ti 12%; per layer 489 us, of which the 5060 Ti branch with its two host-staged hops is 211 us |

3090 kernels 22.2 ms/token (25%), 5060 Ti kernels 2.9 ms (3%), CPU lane busy 54%. Scripts: `bench/glm_nsys_lane_breakdown.py`, `bench/glm_nsys_gpu1_branch.py`.

## 3. Correctness

| Test | Result |
|---|---|
| Determinism of the unchanged build, layered prefill (the fallback path) | **not reproducible**: two identical runs differ, cosine 0.92-0.99, top-1 flips at step 1 |
| Determinism, full-graph prefill (now the GLM default) | **bitwise identical** on all 8 steps |
| New default (graph prefill, no env var) vs the old opt-in | bitwise identical on all 8 steps |
| RAM-tier slab registration fix vs before | bitwise identical on all 8 steps; tracer events 401 -> 0 |
| Planner margin change | bitwise identical on all 8 steps |
| Master after the other agent's environment cleanup (`613d9fe`) vs the earlier build | bitwise identical on all 8 steps |
| 5060 Ti overlap (B) vs master (A) | **bitwise identical on all 8 steps** |

Consequence: the earlier overlap attempt `e45e92d` was reverted for a "parity failure" measured under the layered prefill, which differs from itself by as much (cosine 0.86-0.98); that mismatch was noise.

## 4. Failures found, causes, fixes

| Failure | Cause | Fix |
|---|---|---|
| Every request after the first in a session produced garbage (output `!`) | my own "release the dense host copies after the first decode graph" freed `output.weight`, `attn_k_b`, `attn_v_b`, which the layered paths still read | removed (`b7d6bd1`); benchmarks had sent one prompt per session; any change that frees host memory must be tested with >= 2 prompts |
| 2,313-token prompt aborts: `cudaMemcpyAsync ... invalid argument`, driver: "Copy is larger than memobj size, for destination operand" (also the intermittent `invalid argument` aborts, 3 of ~15 runs) | `ensure_slab_capacity` registered `slot.slab_buffer` outside the mutex and set `slab_pinned` without checking the slot still owned the buffer; a concurrent regrow/release freed the buffer with the registration alive. Tracer (`bench/hostreg_trace.c`): 203 frees of registered memory, 198 refused registrations, all 13.5 MiB slabs | `core/engine.hpp` (`484520b`): register the installed buffer, commit under the mutex, defer frees while a pin is in flight; events 401 -> 0 |
| Same prompt then: GPU `CUDA_ERROR_OUT_OF_MEMORY` in `ggml_cuda_mul_mat_q` (`mmq.cu:257`) | planner margin 0.15 GiB (fitted on AJ) left 0.3-0.45 GiB after the workspace; it completed with 1.45 GiB left | GLM5Next planner margin 1 GiB (`618d454`); other families keep 0.15 GiB |
| GLM graph prefill was opt-in (`ZEROMOE_GLM5NEXT_FULL_GRAPH_PREFILL`) while other families run graph prefill by default | the CUDA MMQ inactive-slot limitation it guarded is fixed by ggml patch 0006 | default on, variable removed (`28114c5`); MTP keeps the layered path |
| Long prompt completes but slowly | cold experts run on the CPU during prefill; `graph_prefill_op_offload = true` runs the 3090 out of memory (peak 24,112 of 24,576 MiB) | open: 2,313 tokens in 408 s = 5.7 tok/s (13-104 token prompts: 20-31 tok/s) |

## 5. Not tested

Qwen family on the slab-pin fix (shared code, adds one short lock per slab registration); the merged tree with 3090 only at pool 88; a held-out ranking check (a survey was ruled out by decision); other GLM files or GPUs for the planner margin; prompts above ~2,300 tokens; MTP (graph path disabled for GLM).
