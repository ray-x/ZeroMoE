# GLM-5.3 decode baseline (RTX 3090 + RTX 5060 Ti, UD-IQ2_M, SSD-offloaded experts)

Recorded 2026-10-20. Reference point for further GLM-5.3 work. Config, scripts and run recipe: `bench/glm53/`.
The full trial log (Trials 1-47, including every negative result) is `docs/perf-baselines.md`.

**Headline:** short-prompt decode **2.00-2.09 tok/s** first-to-last output (1.88-1.96 on the shutdown `Rates:` line), from 1.74-1.78 (Rates 1.65-1.68) with the same recipe and
`MOESTREAM_OFFLOAD_RETARGET=0`. Page-reusing eviction is now on by default whenever SSD offload is active (`docs/offload-retarget.md`). A 300-token generation holds **1.91 tok/s** (Rates). A 426-token prompt decodes at **1.45-1.52 tok/s** (Rates; 1.52-1.59 first-to-last)
with a 74 GiB pool. Prefill 12.4-13.5 tok/s on the 426-token prompt (target 10). Lossless: no expert pruning, same experts computed as before.

## 1. Hardware / software

| Item | Value |
|---|---|
| CPU | Intel Xeon W-2235 @ 3.80 GHz, 6 cores / 12 threads, AVX-512 (VNNI), Skylake-E platform (PCIe Gen3 root ports) |
| RAM | 125 GB (`host_ram_gb = 100`, DDR expert pool 64 GiB; long prompts 74 GiB) |
| GPU 0 | RTX 3090 24 GB: ExpertVault (22 GiB hot-expert set on the second device slot, see `MOESTREAM_GLM5NEXT_GPU1_HOT_GIB`) |
| GPU 1 | RTX 5060 Ti 16 GB: core engine (dense layers, attention, KV, LM head), plus GPU-assist ring |
| Storage | KIOXIA 1 TB (`/data`, offload store) + Samsung 980 PRO 2 TB (model dir); both PCIe Gen3 x4 (platform limit), about 6.5 GB/s combined raw read |
| Model | GLM-5.3 UD-IQ2_M, `/home/ray/vllm-workspace/GLM-5.3-GGUF-align` (6 shards), 78 layers, 75 routed-expert layers |
| Offload store | `/data/glm53_offload.gguf` (routed experts, first layer 3); experts also read from the model files (split reads across both drives) |
| ggml | `ggml-0006-early` with patches `0006` + `0007` (early CPU hook), MoEStream built Release (`build_g7`) |
| Swap | runs launched with `systemd-run --user --scope -p MemorySwapMax=0` |

## 2. Configuration

Config file: `bench/glm53/glm53_dsa_core5060_retarget.toml` (`host_ram_gb = 100.0`, `ram_pinned_budget_gb = 80.0`, `freeze_usage_stats = true`, global_topn pin strategy).
Environment recipe (set by `bench/glm53/run_ab.sh` + `run_dsa_graph2gpu.sh`):

| Variable | Value | Role |
|---|---|---|
| `MOESTREAM_GLM_DSA_FULL_GRAPH`, `_PREFILL` | 1 | full ggml graph for decode and prefill (strict: a failed graph fails the request) |
| `MOESTREAM_GLM_DSA_CPU_LANE` | 1 | CPU lane runs the cold experts while the GPU runs the rest |
| `MOESTREAM_GLM5NEXT_GPU1_HOT_GIB` | 22 | VRAM hot-expert set (23 GiB does not fit; Trial 37) |
| `OFFLOAD_DDR_GIB` | 64 (74 for long prompts) | DDR expert pool |
| `MOESTREAM_GLM5NEXT_PILOT_PREFETCH` / `_STAGE` | 3 / 3 | layer-ahead pilot depth |
| `MOESTREAM_GLM5NEXT_MODEL_DIRECT` | 1 | O_DIRECT reads on the model drive |
| **`MOESTREAM_OFFLOAD_RETARGET`** | **on (default; `0` disables)** | **hand an evicted expert's populated pages to the incoming expert (`mremap(MREMAP_DONTUNMAP)`) instead of `madvise(DONTNEED)` + refault** |
| `MOESTREAM_MEM_GUARD_FLOOR_MB` | 16384 (long prompts only) | lowers the 20 GiB MemAvailable stop floor so a 74 GiB pool fits after a long prefill; default floor unchanged |
| `MOESTREAM_GLM5NEXT_CPU_THREADS` | 6 | lane threads (5-8 flat, 12 is 24% slower) |
| `MOESTREAM_GLM5NEXT_ASSIST_*` | device 0 / second device 1, 3 slots | GPU-assist ring (slot count is flat) |

## 3. Results

### 3.1 Short prompt (22 tokens, 80 generated), 64 GiB pool, interleaved on/off, first-to-last / Rates

| run | RETARGET | first-to-last tok/s | Rates tok/s |
|---|---|---|---|
| rt64 | on / off | 2.03 / 1.75 | 1.91 / 1.66 |
| c_on_a | on | 2.05 | 1.93 |
| c_off_a | off | 1.74 | 1.65 |
| c_on_b | on | 2.06 | 1.92 |
| c_off_b | off | 1.77 | 1.68 |
| clean_on (after debug-code cleanup) | on | 2.09 | 1.96 |
| clean_off | off | 1.78 | 1.68 |
| memprof (memory sampling run) | on | 2.00 | 1.88 |

Same-config spread is about +-0.05. Per-step view of rt64: mean 453 ms (2.21 tok/s) with retarget vs 532 ms (1.88 tok/s) without; fastest single step 305 ms (3.28 tok/s) vs 414 ms (2.41 tok/s).
Lane breakdown, ms/token, retarget on vs off: begin_load 14 vs 48, lane wait 26 vs 62, split stall 139 vs 173, compute 351 vs 394, read 4.7 vs 5.4 ms per load.

### 3.2 Sustained generation (22-token prompt, 300 generated), retarget on

1.91 tok/s Rates (1.94 first-to-last, 1.80 overall). Lane stall flat at about 178 ms/token to the end; 0 failed page moves (35,073 full + 4,686 mixed-size moves); output coherent, no repetition.

### 3.3 Long prompt (426 tokens, 80 generated), retarget on, guard floor 16 GiB, `host_ram_gb = 100`

| Pool | Rates tok/s | first-to-last | notes |
|---|---|---|---|
| 74 GiB, run 1 | 1.45 | 1.52 | prefill 13.4 tok/s |
| 74 GiB, run 2 | 1.52 | 1.59 | |
| 78 GiB | - | - | refused: MemAvailable under the 16 GiB floor at decode-graph build |
| 74 GiB, retarget off (Trial 41) | 1.31 | - | for comparison |

### 3.4 Progression (22-token prompt, Rates line unless noted)

| Step | tok/s |
|---|---|
| layered path before row-LCM stride fix | 0.38 |
| after stride fix | 0.99 |
| full graph, CPU lane, split reads, 22 GiB hot set (Trial 19-21) | 1.43 |
| + pilot 3/3 + direct model reads | 1.50-1.54 |
| + graph LM head (silent CPU fallback removed, about 50 ms/token) | 1.62-1.64 (first-to-last 1.71-1.73) |
| + page-reusing eviction (`RETARGET=1`) | 1.88-1.96 (first-to-last 2.00-2.09) |

## 4. Memory at peak performance

Sampled during steady decode at token 42 of the `memprof` run (retarget on, 64 GiB pool, 22 GiB hot set, 2.00 tok/s first-to-last):

| Item | Value |
|---|---|
| Process RSS | 90.35 GB (anonymous 89.19 GB, file-backed 0.19 GB, shared 0.97 GB); 86.3 GB at shutdown |
| mlocked | 0.47 GB |
| System MemAvailable | 32.8 GB (MemFree 28.9 GB, page cache 6.0 GB, AnonPages 90.7 GB, Mapped 1.3 GB) |
| Host ledger (graph/KV/resident tensors, excludes the DDR pool) | 19.0 GB of the 100 GB `host_ram_gb` |
| VRAM, RTX 3090 | 23,720 MiB of 24,576 MiB used (22 GiB hot set) |
| VRAM, RTX 5060 Ti | 14,732 MiB of 16,311 MiB used |
| Memory mappings (VMAs) | 15.1k at token 42; 35.7k at token 104 of the 300-token run (bounded by one mapping per layer/expert/plane, about 115k worst case; limit 1,048,576) |
| Threads | 30 |

Long prompt, 74 GiB pool: MemAvailable 22.5 GB at decode-graph build (15.0 GB low-water at 78 GiB, where the build is refused), RSS 100.9 GB at shutdown.
The roughly 33 GB shown as available during a short-prompt run is the guard reserve (stop floor 20 GiB, 16 GiB with the override), pinned/driver memory outside the host ledger, and
reclaimable cache; memory-mapping the dense weights (Trial 42) freed 10 GiB of anonymous memory but did not raise the usable pool ceiling.

## 5. Correctness

- Greedy (temperature 0) output with retarget on equals retarget off for the first 318-410 characters of about 420; one on/off pair is identical. Two retarget-off runs diverge from each other at the
  same late point (near-tie routing; ms/token and text vary run to run). The 300-token output is coherent.
- Retarget moves only already-populated pages between slots that the cache has just evicted, under the arena lock; a failed `mremap` falls back to freeing the victim's pages.

## 6. What does not help (measured, see `docs/perf-baselines.md`)

Prefetch depth (3/3 is best, 8/8 is -17%), wider or confidence-based pilot (Trial 44, slower), demand-priority gate for reads (39), pool size beyond 64 GiB on short prompts (40),
OMP binding/wait policy/thread count (26-27), IQ3 AVX-512VL dot (25, +1%), ranking refit (22, 38b), prompt-adaptive resident set oracle (+7.6% at best, 43), mapping dense weights (42),
asynchronous eviction (46), balanced drive split (-12%), larger 3090 hot set (does not fit), Gen4 link for the Samsung drive (platform is Gen3).

## 7. Reproduce

    bench/glm53/run_ab.sh <tag> MOESTREAM_GLM5NEXT_GPU1_HOT_GIB=22 OFFLOAD_DDR_GIB=64 NGEN=80 \
      MOESTREAM_GLM5NEXT_PILOT_PREFETCH=3 MOESTREAM_GLM5NEXT_PILOT_STAGE=3 \
      MOESTREAM_GLM5NEXT_MODEL_DIRECT=1 # MOESTREAM_OFFLOAD_RETARGET is on by default

Check `nvidia-smi`/`pgrep` first (`bench/glm53/preflight.sh` does it), run under `systemd-run --scope -p MemorySwapMax=0`, and judge text coherence as well as tok/s.
