# GLM-5.3-Flash: configuration, improvements and test data (handoff, 2026-10-11)

Everything below is for GLM-5.3-Flash on this box: Xeon W-2235 (6C/12T, AVX-512 VNNI), 125 GB DDR, RTX 3090 24 GB (PCIe 4) for all
numbers unless marked, RTX 5060 Ti 16 GB (PCIe 3) only where stated, two NVMe drives (btrfs model drive, XFS second drive).
Rules that held for every number: 3090 only unless stated, nvtop/htop closed, no compile jobs alongside (arms are retried when contaminated),
output read for coherence before a timing was trusted, page cache dropped before each arm.

## 1. The configuration we run

**Model:** `/home/ray/vllm-workspace/GLM-5.3-Flash/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf` (+ shard 2), 87.3 GB, IQ2_XXS
(2.06 bpw; 8 layers have down in IQ2_S). `glm5-next 320B.A18B`: 45 trunk layers (34 KDA + 11 DSA), dense FFN layers 0-2, MoE layers 3..44
(42 layers) x 288 routed experts, top-8 + shared expert, hyper-connections hc_mult=4. Routed experts total 74.21 GiB = 12,096 experts of
~6.6 MB (planes 2,162,688 / 2,162,688 / 2,686,976 B). Always pass the explicit first-shard path (a directory merges sibling .gguf files).

**Engine setup (test harness `glm_runs/run_seg_ab.sh`, config `glm_runs/spread_off.toml`):**

| item | value |
|---|---|
| decode | full graph (default; `MOESTREAM_GLM5NEXT_FULL_GRAPH=0` opts out, `GRAPH_STRICT=1` fails a declined graph) |
| placement | "spread": VRAM holds every layer's hottest experts (`moe_cache_gb=15.8`, `dense_gb=6.5`, `vram_cache_full_layer_prefix=0`, `pin_strategy=global_topn`, `pin_topk_per_layer=12`, `pinned_vram_capacity=2400`); all layers 3..44 offloaded (`OFFLOAD_FIRST_LAYER=3`) |
| DDR MoE pool | `MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB` (virtual arena; 32 default in tests; 46 for a 64 GiB machine) |
| cache policy | `OFFLOAD_POLICY=lru`, `OFFLOAD_PINNED=0.5` (half the pool pinned by usage ranking), `OFFLOAD_ASYNC=1` |
| prefetch | off (`PILOT_STAGE` unset) |
| store | embedded expert blocks in an interleaved model, or a store striped over both NVMes: `OFFLOAD_STORE=/data/moestream-offload/glm53flash-stripe0.gguf:/home/ray/work/nvme1-store/glm53flash-stripe1.gguf` |
| usage file | `docs/ssd-offload/data/glm53flash_usage_combined_old_plus_survey5.txt` |
| other | `host_ram_gb=16` (>= ~16 needed for the dense tensors), `THP=0`, `CPU_THREADS=8`, `ALLOC_FLOOR_MIB=512`, `CUDA_VISIBLE_DEVICES=0`, `--ngen 96`, 5 topic-shifting prompts in one session |
| older whole-layer baseline | `configs/glm53flash_AJ_IQ2XXS_wholelayer8_cpu.toml` (8 MoE layers fully in VRAM, rest on CPU), 7.26 tok/s |

Layout/tools: `gguf_tool pack --expert-interleave` (4 KiB-aligned embedded `offload.blk.*` blocks, one preadv per expert; the engine reads
the original GGUF or the packed one), `gguf_tool offload --expert-mod/--expert-rem` (store striped by expert parity over several drives;
`OFFLOAD_STORE` and `OFFLOAD_MODEL` take file, directory or `a:b` lists).

## 2. Improvements made (GLM path)

| area | what | effect |
|---|---|---|
| decode graph | LM head in graph, whole-layer VRAM fix, fused hyper-connection ops (31k -> 6k nodes), GPU1 spill | 2.70 -> 6.87 tok/s (parity with neurall 6.99) |
| spread placement | hot experts of every layer in VRAM instead of whole layers | 7.0 -> 8.1 tok/s no offload |
| SSD offload in graph | virtual arena slot = expert id, CPU miss node before MUL_MAT_ID, bit-exact (0.000 logit delta) | runs 74 GiB of experts on 32 GiB DDR |
| second GPU hot branch | 14.2 GiB of next-hottest experts on the 5060 Ti | 125 -> 100.6 ms/token at 32/48 GiB DDR (dual GPU only) |
| THP on arena | huge pages | -22 ms/token CPU TLB cost |
| async loads | `begin()/finish()` split, loading state, no lock across reads | prefetch regression 4% -> 0 |
| layout | one O_DIRECT preadv per expert from embedded blocks | 229.7 -> 219.7 ms/token (-4.4%) |
| striping | store striped over both NVMes | 219.7 -> **204.7 ms/token (-10.9% vs original)** |
| boot seeding | 8 reads in flight (`SEED_PAR`) | 12.5-18.5 s -> 6.5 s |
| bugs | stale pilot snapshot (recall 37 -> 71%), multi-turn decode-state clobber (later turns answered turn 1), prefetch lock held across reads, stale probation entries, ridge buffers below the 0.5 GiB alloc floor (silent legacy fallback), repetition-penalty prompt scan | see docs 05, 09 |

## 3. Test data

### 3.1 Pool / DDR size (striped store, LRU, 50% pinned, no prefetch, 5 rounds / 475 tokens, outputs identical)
| DDR pool | pool experts | machine / cgroup | demand loads | miss-node time | ms/token | tok/s |
|---|---|---|---|---|---|---|
| 24 GiB (3090+5060, earlier) | - | any | ~17 miss/token | - | 161 | 6.2 |
| 32 GiB | 5,215 (43%) | any | 14,639 | 32.4 s | 204.7 | 4.89 |
| 46 GiB | 7,497 (62%) | 64G | 4,838 | 10.5 s | 149.1 | 6.71 |
| 58 GiB | 9,453 (78%) | 100G | 179 | 1.1 s | 125.1 | 7.99 |
| 100 GiB (capped at 74.2 GiB = all experts) | 12,096 | 110G | queued, not yet measured | | | |

The first row is the older dual-GPU spread run (`05-glm-full-graph-integration.md`); the 32-58 GiB rows are the 3090-only striped-store runs.
A 64 GiB machine fits a 46 GiB pool (RSS = pool + ~17 GiB); 58 GiB needs ~75 GiB.

### 3.2 Layout and striping (DDR 32 GiB, same traffic, 14,639 demand loads in every arm)
| arm | ms/token | reads |
|---|---|---|
| original shards + cold3173 store | 229.7 | 5,101 direct + **14,753 buffered** |
| interleaved model, embedded blocks | 219.7 | 19,854 direct, 0 buffered |
| store striped over two NVMes | **204.7** | 19,853 direct, 0 buffered |
| aligned-only model (no per-expert block) with old store | 243.1 | buffered reads dominate; alignment alone buys nothing |

H2D lane facts: pinned DMA 12.2 GB/s (3090) / ~6.9 GB/s (5060 Ti); 3 DMAs vs 1 differ by 2.5%; a bounce copy costs ~120 us per expert;
fixed staging overhead 0.5 ms/layer. CPU wins unless an uploaded expert is reused about twice or more.

### 3.3 Prefetch (DDR 32 GiB, 50% pinned, 5 rounds, 475 tokens)
| arm | demand loads | prefetch loads | ms/token |
|---|---|---|---|
| lru, no prefetch | 14,639 | - | 232.0 |
| lru + prefetch K=4, lock held | 8,877 | 10,646 | 241.4 |
| lru + prefetch K=4, **async** | 9,094 | 10,428 | **231.6 / 232.3** |
| async, lookahead k=2 / k=3 | 10,335 / 11,288 | ~10.7k | 236.1 / 239.5 |
| async, K=8 / K=2 | 7,121 / ~11.7k | 24,254 / - | 243.5 / 240.6 |

Pilot precision by rank, k=1: 92 / 80 / 65 / 51 / 40 / 29 / 22 / 18% (r1..r8). Ridge correction: +4..13 points offline, only +1..2 live
(not recommended). Conclusion: prefetch is neutral at best; the token is CPU cold-expert compute plus synchronous loads, and more DDR is the
lever that works.

### 3.4 Cache policy / LRU strategy (same setup)
| arm | demand loads | ms/token |
|---|---|---|
| lru | 14,639 | 232.0 |
| segmented LRU quota 0.25 / 0.0 / 0.02 (locked) | 10,189 / 11,266 / 8,735 | 252.8 / 255.6 / 242.8 |
| segmented 0.02, async | 8,966 | 231.6 / 232.7 |
| pinned fraction 0.9 / 0.5 / 0.25 / 0.0 (homogeneous traffic) | 15,147 / 11,413 / 10,925 / 10,293 | more LRU is better when ~54% of the offloaded experts fit |
| dynamic VRAM promotion (`DYNAMIC_CACHE`, `DYNAMIC_HEAT`) | - | no-op: aliased path hard-sets `dynamic_expert_cache=false` (arms 196-198 ms = noise) |

Private DDR tier can be worse than the page cache when the model fits in RAM (host_ram_gb 95 -> 24: +34% prefill, Qwen). On this box
zram swap and `host_ram_gb` too close to RAM caused fault storms (see memory notes).

### 3.5 Quant types
**Measured: only AJ-IQ2_XXS.** Other GLM-5.3-Flash files on the box (UD-IQ1_S 93 GB, UD-IQ3_XXS partial, GSQ-RCO-3.0bit 117 GB) were not
benchmarked here. The modelled table (`08-quant-capacity-model.md`; fitted to the AJ runs at DDR 24/32/48 GiB = 161/128/100 ms, error <= 1 ms;
static placement, hypothetical uniform quant, accuracy NOT modelled) for 3090 with 28.9 GiB VRAM experts:

| quant | MB/expert | DDR 32: ms/token | DDR 64: ms/token | CPU GB/s |
|---|---|---|---|---|
| iq1_m | 4.1 | 62 | 62 | 11.3 |
| iq2_xxs | 4.9 | 72 | 72 | 11.8 |
| q2_0 | 5.9 | 88 | 81 | 14.8 |
| iq3_xxs | 7.2 | 156 | 104 | 13.6 |
| q3_k | 8.1 | 175 | 78 | 32.7 |
| iq4_xs | 10.0 | 308 | 126 | 37.0 |
| q4_0 / q4_k | 10.6 | 354 / 350 | 152 / 148 | 37.5 / 41.8 |
| q8_0 | 20.1 | 1224 | 840 | 48.0 |

CPU cost per expert (measured, `07-cpu-cost-by-quant.md`): 1-3 bit codebook types are compute-bound (11-15 GB/s; IQ3_XXS costs 1.8x
Q2_0 and 2.2x Q4_0 per expert), 4-8 bit types are memory-bound (33-48 GB/s). Q2_0 VNNI verified 2.3x decode / 2.8x prefill (Qwen).
The real AJ file is 6.59 MB/expert, so its measured 128 ms at DDR 32 is slower than the uniform iq2_xxs row.

### 3.6 Other measured facts
* GPU utilisation during offload decode ~7%; miss node ~37% of decode time at 32 GiB.
* Refault after MADV_DONTNEED: 0.59 ms CPU / 3.08 ms wall per 7 MB expert; `OFFLOAD_EVICT=free` removes it (opt-in).
* GPU utilisation balance 3090 vs 5060 Ti, cuFile/GDS (btrfs not supported), io_uring batching: closed, see memory index.

## 4. Where we stand and what is open
* Best measured single-3090 GLM: **8.0 tok/s (125 ms/token) at a 58 GiB pool**; neurall's llama.cpp on the same file 7.37 (whole-layer baseline 7.26).
* A 20 tok/s report (ExLlama) is not reproducible from our data: it needs ~50 ms/token, i.e. the CPU cold-expert compute and graph time
  must fall by 2.5x. Unknown: that report's model/bpw/context/placement. Next step is per-layer timing at 58/100 GiB to split the 125 ms.
* Open: dynamic VRAM promotion needs a topology retarget-slot op (doc 10 section 6); stripe-aware boot seeding; ModelReader O_DIRECT for
  aligned shards; Qwen IQ3_S stripe/interleave stores; measure the 100 GiB pool and real UD-IQ1_S/UD-IQ3_XXS runs to replace the modelled quant rows.

## 5. Where things live
* Branch `prefetch-prediction` (worktree `~/work/moestream-prefetch`, build `build_pf`); tools and layout on master f4dcc1a.
* Docs: `docs/ssd-offload/05` (graph integration), `07` (CPU cost), `08` (quant model), `09` (summary), `10` (layout, lane, pool, seeding).
* Drivers: `glm_runs/run_seg_ab.sh`, `run_clean.sh`, queue scripts `run_*_queue.sh`; reports `glm_runs/pinab_report.py`; logs under
  `~/work/moestream-graph/glm_runs/pinab/` (the toml's absolute `log_path` points there).
* Stores: `/data/glm53flash-il` (interleaved), `/data/moestream-offload/glm53flash-stripe0.gguf` + `~/work/nvme1-store/glm53flash-stripe1.gguf`
  (striped, 37.11 GiB each), `glm53flash-aj-iq2xxs-all12096.gguf` (single full store, 80 GB).

## 6. Update 2026-10-13: decode profile and two measured wins (graph-framework branch)

Decode profile at DDR 32 GiB, lru, 3090 only (`bench/glm_decode_profile.sh`; ~230 ms/token before this work):
CPU cold-expert MUL_MAT_ID (IQ2_XXS) ~93 ms, miss node (SSD reads) ~70-90 ms, GPU ops ~45 ms, readbacks 0.13 ms,
cache update 0 ms. H2D/D2H transfers are not a lever here; the token is CPU compute plus synchronous loads.

| change | where | decode tok/s (order-balanced, logits+tokens identical) |
|---|---|---|
| lane computes resident cold experts while missing ones load (`MOESTREAM_GLM5NEXT_SPLIT_LOAD`, default on) | `include/early_ready_lane.hpp`, `ExpertStore::begin_ensure/finish_ensure` | 4.29 / 4.28 -> 4.92 / 4.92 (+14.8%) |
| IQ2_XXS x Q8_K dot, per-block vector sign/scale expansion (ggml fork branch `iq2xxs-avx512vl-dot`) | `ggml/src/ggml-cpu/arch/x86/quants.c` | 4.94 / 4.93 -> 5.27 / 5.23 (+6.7%) |

After both: lane compute ~106 ms/token, residual load stall ~42 ms/token (about half of all layer calls split),
id wait ~25 ms/token. Gathers (`vpgatherdq`) run 3-5x slower than scalar loads on this Cascade Lake (GDS mitigation):
`bench/kernels/iq2xxs_dot_bench.cpp` has the exactness check and the rejected variants.
`bench/glm_env_ab.sh` runs any two env/library setups order-balanced and compares logits + tokens.
