# Ling 3.1 Flash native-cache graph (2026-10-19)

## Implementation

Ling 3.1 Flash (`bailingmoe3`) has an opt-in full-graph path through
`MOESTREAM_LING_FULL_GRAPH=1 MOESTREAM_LING_NATIVE_GRAPH=1`. It builds one
CPU callback node per layer and delegates attention, dense FFN, and MoE work
to the existing `FamilyOps`, `dense_ffn_gguf`, and `moe_forward_gguf` paths.
The graph owns no expert storage. RAM and VRAM expert ownership stays with
`RAMSafetyManager` and `GpuTopology`, as it does for the DeepSeek-V4.1
cache-backed graph. Both prefill (`S=2`) and decode (`S=1`) built 72-node,
single-split graphs.

Device placement is not fixed by model layer. The two GPU cache budgets and
active-pool weights select expert IDs dynamically; the same layers can have
experts resident in DDR, the 3090, or the 5060 Ti. The usage profile ranks
expert IDs and does not bind layer ranges to devices.

Graph startup now treats this Ling path as a native-cache graph: it retains
the usage-ranked preload and uses the existing dense-weight pool instead of
allocating a graph-owned copy. The older GLM-derived graph remains available
when `MOESTREAM_LING_NATIVE_GRAPH` is unset. Ling's legacy partial graph also
keeps batched MMQ negative-route skipping disabled because that policy
regressed this model's short prefill; the single-token MMVQ skip remains.

No CUDA kernel source changed in this work. The engine was rebuilt to compile
the C++ graph adapter and cache integration; the graph invokes existing
attention and MoE kernels through native callbacks.

## Initial 80 GiB benchmark configuration

- Model: `/home/ray/vllm-workspace/ling-3.1-flash/Ling-3.1-flash-AD-IQ1_S-00001-of-00004.gguf`
- Usage profile: `/tmp/ling-3.1-flash-usage-fullgraph.pin` (5,879 expert records)
- Input: raw `Hi there` (2 tokens); 32 output tokens; context 64; greedy
- Host budget: 80 GiB DDR, with the engine's 21.5 GiB startup margin
- Per-process cap: `MemoryMax=90G`, `MemorySwapMax=0`; startup saw 117.0 GiB
  available and required 101.5 GiB
- RAM tier: automatic byte-aware capacity of 16,184 expert slots / 57.3601
  GB. Preload filled all 16,184 slots; occupancy was 57.3057 GB after prefill
  and 56.9595 GB after decode. RAM pin budget was 15 GB.
- GPU main caches: configured 8 GB on the 3090 and 13 GB on the 5060 Ti.
  The engine allocated 4.57638 GB on the 3090 and 13 GB on the 5060 Ti
  (17.5764 GB total). The 3090's 12 GB dense allocation limits its expert
  cache.
- Dynamic active-expert pools: 24 slots / 1.2729 GB on the 3090 and 8 slots
  / 0.4243 GB on the 5060 Ti. These are per-device cache buffers, not
  layer-to-GPU assignments.
- Usage preload: DeepSeek-style global top-N, 2,048 pinned VRAM experts,
  1,224 unpinned VRAM experts, and 16,184 RAM experts. Prefill warm-up via
  the shared prefetcher, adaptive pilot prefetch, 10 expert-fetch threads,
  6 prefetch workers, 10 OMP threads, and 8 attention-score threads.
- `io_uring` was enabled. The GGUF declares 32-byte alignment, so
  `O_DIRECT` was disabled and reads used the page cache.

## Result and limits

The initial 80 GiB run built both graph shapes and produced:

- Prefill: 2 tokens in 1.40983 s, **1.42 tok/s**
- Decode: 32 tokens in 18.525 s, **1.73 tok/s** request wall rate
- First-to-last output interval: **1.69 tok/s**
- Overall request rate: **1.60 tok/s**
- Reported RSS at shutdown: **73.01 GB**
- Sampled cgroup usage during preload: **78.4 / 90 GB**, with
  `memory.swap.current=0`
- Expert residency at shutdown: **56.96 GB DDR** and **15.08 GB** in the
  persistent GPU expert caches; both GPUs had active cache allocations.

The best run is still below the reported DeepSeek-V4.1 reference of 7 tok/s
decode and 20 tok/s prefill. The two-token prefill is particularly sensitive
to fixed request and graph setup costs, so its rate is not directly
comparable to a longer prompt. `nvidia-smi` worked during admission checks
but intermittently failed after inference; per-device allocations above
come from the engine's own cache logs, not a sampled peak of total board
memory. During sampled execution the benchmark cgroup reported
`memory.swap.current=0`; host-wide zram had other usage.

The same-prompt cache experiments progressed from 0.83/0.34 tok/s
(decode/prefill, no usage preload) to 1.62/0.72 with full DDR preload,
1.67/1.13 with asynchronous prefill warm-up, and 1.73/1.42 with the
DeepSeek-style global top-N and adaptive cache settings. These results
validate the native-cache graph integration, but do not meet the 10/20
performance target. The remaining gap requires work in Ling's attention and
expert execution path, beyond cache sizing alone.


## Latest DDR-capacity follow-up

The same config and prompt were rerun with a host RAM budget of 90 GiB,
`MemoryMax=100G`, and `MemorySwapMax=0`. Startup admitted 117.3 GB available
against a requirement of 111.5 GB. The byte-aware DDR pool grew to 19,040
expert slots / 67.4824 GB; it held 66.9739 GB after decode. The run evicted
1,464 DDR experts during decode, down from 1,844 with the 80 GiB pool. A
sampled cgroup reading was 84.9 / 100 GiB with `memory.swap.current=0`.

This is the best same-prompt result so far:

- Prefill: 2 tokens in 1.34774 s, **1.48 tok/s**
- Decode: 32 tokens in 18.108 s, **1.77 tok/s** request wall rate
- First-to-last output interval: **1.72 tok/s**
- Overall request rate: **1.64 tok/s**
- Reported RSS at shutdown: **82.15 GB**
- Expert residency at shutdown: **66.97 GB DDR** and **15.15 GB** in the
  persistent GPU expert caches

The change from the 80 GiB pool was small: about 2% decode and 4% prefill.
Cache sizing and preload policy are no longer the main source of the gap to
DeepSeek-V4.1. The request profile still attributes significant time to RAM
lookups and reads, Ling attention, and the output projection. The target is
still unmet.
