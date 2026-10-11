# GGUF layout for expert reads, and the CPU-vs-GPU lane decision (2026-10-11)

## 1. Layout: what `gguf_tool pack --expert-interleave` does
* `pack --align 4096` keeps the packed `blk.L.ffn_{gate,up,down}_exps.weight` tensors and re-bases every tensor on a 4 KiB boundary (this alone
  removes the engine's "not 4096-aligned" warning and the buffered fallback).
* `--expert-interleave` (align-only mode) additionally writes, per MoE layer, one tight `[gate|up|down]` block per expert under the offload-store
  names `offload.blk.<L>.ffn_<k>.<E>.weight`, each block starting on `--align`, plus the marker KV `moestream.expert_interleave=1`. The packed
  tensors are untouched, so every existing loader keeps working; `OffloadStore::open_embedded/open_auto` read the blocks straight from the model
  shards (several files, directory accepted) with one `preadv(O_DIRECT)` per expert. Output grows by the routed-expert bytes.
* Mixed quant types need nothing special: each plane keeps its own byte size (GLM-5.3-Flash: 2,162,688 / 2,162,688 / 2,686,976 B), and the
  blocks of one layer never straddle shards. Pass the FIRST shard file of a multi-shard model (the loader pulls in the siblings; directory mode
  would write the whole model once per shard).
* GLM-5.3-Flash AJ-IQ2_XXS: 87.3 GB in, 167.0 GB out (3 shards, 37,671 tensors), 600/600 sampled tensors byte-identical.

## 2. NVMe read cost by layout (`bench/expert_layout_read_bench.cpp`, O_DIRECT, random experts, 6.6 MB each)
| threads | packed (3 reads) | embedded (1 preadv) |
|---|---|---|
| 1 | 354 experts/s, 2.82 ms | **427 experts/s, 2.34 ms** (-17% latency) |
| 4 | 427 experts/s (2.82 GB/s) | 465 experts/s (3.07 GB/s) |
| 8 | 427 experts/s | 465 experts/s |
The drive saturates at ~3.1 GB/s, so the layout wins mostly at queue depth 1 (the latency of a single demand miss); throughput is bounded by
the device. More bandwidth needs a second NVMe (striping), not a different layout.

## 3. CPU lane vs upload-and-compute-on-GPU (`bench/lane_decision_model.py`)
Inputs measured on this box: CPU cost per expert by quant (07-cpu-cost-by-quant.md), pinned H2D 12.2 GB/s on the 3090 (PCIe 4) and ~6.9 GB/s on
the 5060 Ti (PCIe 3), 3 DMAs vs 1 differs by 2.5%, extra graph/scheduler overhead of the staging nodes 0.5 ms/layer, +3.5% CPU while a DMA runs.

| quant | MB | CPU us/expert | DMA us (3090) | CPU:DMA time ratio (3090 / 5060 Ti) |
|---|---|---|---|---|
| q4_0 / q3_k / q4_k / iq4_xs | 2.5-3.3 | 74-85 | 214-279 | 0.28-0.36 / 0.16-0.21 |
| q2_0 | 1.38 | 94 | 120 | 0.78 / 0.45 |
| iq2_xxs / iq2_xs / iq3_xxs | 1.5-2.3 | 129-166 | 132-192 | 0.86-0.98 / 0.50-0.57 |
| GLM iq2_xxs (4.87 MB) | 4.87 | 390 | 406 | 0.96 / 0.55 |

A ratio below 1 means the CPU computes the expert faster than it can be uploaded: the upload lane only helps when CPU cost per byte is as high as
the link is slow, i.e. the compute-bound codebook types (IQ1-IQ3, 11-15 GB/s effective) on a fast link. On the 5060 Ti the link is 1.8x slower, so
the CPU wins for every type. Break-even per layer on the 3090 (pinned arena, no bounce copy): `K * p * t_cpu > 0.5 ms + 0.035 * cpu_layer`.
GLM IQ2_XXS with 3-5 cold routes needs K >= 1.5-2.2 staged experts at precision 0.9-0.65; IQ3_XXS needs K >= 3.5; Q2_0 and the K-quants
never repay. This matches the measurements: K=1 staging was neutral (-1%), K>=2 CPU-copy bound; async prefetch alone neutral.

Decision rule used from now on: keep cold experts on the CPU; consider the upload lane only for IQ1-IQ3 quants, a PCIe 4 card, a pinned
(bounce-free) arena and a pilot precision >= 0.8 at the staged ranks; the first thing to remove is the 0.5 ms/layer fixed overhead.

## 4. End to end (GLM-5.3-Flash, DDR 32 GiB, 50% pinned, async lru, no prefetch, 5 topic-shifting rounds, 475 decode tokens, clean arms)
Identical traffic in every arm (14,639 demand loads); outputs identical to the original.

| arm | ms/token | miss-node load time | reads served from | boot seeding |
|---|---|---|---|---|
| original shards + cold3173 store | 229.7 | 41.4 s | store direct 5,101; **main GGUF buffered 14,753** | 20.0 s |
| interleaved model, embedded blocks | 219.7 (-4.4%) | 38.6 s | embedded direct 19,854; buffered 0 | 14.9 s |
| store striped over two NVMes (even/odd experts) | **204.7 (-10.9%)** | 32.4 s | striped store direct 19,853; buffered 0 | 19.7 s |

Three quarters of the reads in the original setup were buffered per-plane reads of the main GGUF (the cold store only covers 3,173 experts).
One direct read per expert recovers 4.4%; spreading the store over both NVMes (6.7 GB/s aggregate vs 3.1) recovers another 6.8%.
Controls from the same queue: a repeat of the original arm gave 234.1 ms/token but was flagged contaminated (82 samples with another process running), so it is NOT a clean estimate of run-to-run spread, and the interleaved
(4 KiB-aligned) model with the default cold store, whose remaining reads are still buffered, gave 243.1 ms: alignment of the model file
alone buys nothing while the model reader uses buffered per-plane preads (14,753 buffered reads); the gain comes from reading each expert
with one direct preadv from a store that covers every expert, and from spreading that store over both NVMes.

## 5. Pool size (striped store, async lru, no prefetch, same 5 rounds / 475 tokens, outputs identical)
| DDR MoE pool | pool experts | machine | demand loads | miss-node load time | ms/token | tok/s |
|---|---|---|---|---|---|---|
| 32 GiB | 5,215 (43% of 12,096) | any | 14,639 | 32.4 s | 204.7 | 4.89 |
| 46 GiB | 7,497 (62%) | cgroup 64G (RSS = pool + ~17 GiB) | 4,838 | 10.5 s | **149.1** | **6.71** |
| 58 GiB | 9,453 (78%) | cgroup 100G | 179 | 1.1 s | **125.1** | **7.99** |
A 64 GiB machine fits a pool of about 46 GiB: 14,639 -> 4,838 demand loads and -27% ms/token. 58 GiB would need a ~75 GiB machine (host_ram_gb is the process-wide
host cap: 2 and 6 GiB refuse the dense tensors and produce empty outputs; 16 GiB works). At 58 GiB almost every miss is gone; the remaining 125 ms/token
is CPU cold-expert compute and the graph, which prefetch and read layout cannot touch.

Dynamic VRAM promotion (MOESTREAM_GLM5NEXT_DYNAMIC_CACHE, DYNAMIC_HEAT) is a no-op in this configuration: the aliased-VRAM path hard-sets
dynamic_expert_cache=false, so those arms (196.5-198 ms) are run-to-run noise (about 4%).
