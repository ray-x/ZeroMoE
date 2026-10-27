# Ling 3.1 Flash native-cache graph (2026-10-19)

## Implementation

Ling 3.1 Flash (`bailingmoe3`) runs the full graph by default; the
cache-backed implementation is selected with `ZEROMOE_LING_NATIVE_GRAPH=1`
(the old `ZEROMOE_LING_FULL_GRAPH` gate was removed). It builds one
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
when `ZEROMOE_LING_NATIVE_GRAPH` is unset. Ling's legacy partial graph also
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

## MLA GPU fast-path follow-up

With the same 90 GiB DDR configuration, enabling `attn_qabsorb_gpu` with a
1 GiB cap uploaded the BF16 WUK cache for 9 of 72 layers (0.0755 GB; the
remaining layers lacked those weights or did not fit the live VRAM budget).
The run measured **1.81 tok/s decode**, **1.51 tok/s prefill**, **1.77 tok/s
first-to-last output**, and **1.68 tok/s overall**, with 82.23 GB reported
RSS. Sampled cgroup usage remained below 89 GB / 100 GB and
`memory.swap.current=0`. This is the current best same-prompt result, a small
gain over the 90 GiB baseline.

The separate MLA score GPU trial reached 1.80 tok/s decode but slowed the
short prefill to 1.39 tok/s. Forcing legacy CPU score during prefill restored
1.48 tok/s prefill but measured 1.76 tok/s decode, so this score path is not
part of the current best config. Neither attention fast-path trial closes
the 10/20 target.


## Attention-path A/B and dispatch diagnostic

With the same 90 GiB DDR pool, enabling the MLA Q-absorb GPU path (1 GiB
cap; score GPU path off) uploaded WUK weights for 9 of 72 layers using
0.0755 GB. It measured **1.81 tok/s decode** and **1.51 tok/s prefill**,
which remains the best normal-log-level result. Enabling the MLA score GPU
path alone measured 1.80 / 1.39 tok/s (decode/prefill); forcing legacy CPU
score for prefill measured 1.76 / 1.48. Neither score-GPU variant was kept
in the test config.

A DEBUG diagnostic confirmed that the 1.97 GB “embedding + LM head” dense
category is part of the 3090's GPU dense pool; it did not log the CPU
`dense_proj` fallback for `output.weight`. The reported ~4.1 s LM-head timer
therefore includes the GPU dispatch path and may also include synchronization
with earlier asynchronous work. The next execution-path investigation must
separate that wait from the output projection itself. The diagnostic run was
not used as a throughput result.

## Hybrid CUDA graph A/B (2026-10-19)

The opt-in Ling hybrid path now builds the KDA/MLA attention graph while each
MoE layer calls the existing native expert cache. It reuses the engine's dense
weight pool and keeps the usage-ranked DDR/VRAM cache ownership. The graph
borrows the host-resident MLA WUK/WUV tensors through `resident_tensor`; this
avoids loading them into the 1 MiB native dense marker arena. No CUDA kernel
source changed.

The 32-token A/B used the same raw two-token `Hi there` prompt that generated
the fixed usage profile, `/tmp/ling-3.1-flash-usage-fullgraph.pin` (5,879
records). The profile was left unchanged. Context was 64, generation was
greedy, and the opt-in selectors were `ZEROMOE_LING_NATIVE_GRAPH=1` and
`ZEROMOE_LING_HYBRID_CUDA_GRAPH=1` (the family-wide full-graph gate no longer
exists).

The test used `host_ram_gb=90`, `MemoryMax=100G`, and `MemorySwapMax=0`.
Startup admitted 116.9 GB available against the engine's 111.5 GB requirement.
The GPU cache budget was moved from the 3090 to the 5060 Ti to leave graph
workspace on the core device: 3.0 GB on the 3090 and 14.4 GB on the 5060 Ti.
The two-GPU active pools remained 24 slots / 1.2729 GB and 8 slots / 0.4243 GB.
The engine logged 17.753 GB of dense weights on the 3090 and 37.66 GB total
allocated across both devices after cache setup. Expert residency at shutdown
was 66.7741 GB in DDR and 15.1398 GB in the persistent GPU caches.

Both graph shapes built successfully:

- Prefill: 72 layers, 4,073 nodes, 172 splits, 0.27 GiB workspace.
- Decode: 72 layers, 4,139 nodes, 173 splits, 0.27 GiB workspace.

The request measured:

- Prefill: 2 tokens in 1.51866 s, **1.31695 tok/s**.
- Decode: 32 tokens in 12.834 s, **2.49 tok/s** request-wall rate and
  **2.44 tok/s** first-to-last output rate.
- Overall: 32 generated tokens in 14.36 s, **2.23 tok/s**.

The expert cache reported 7,532 hits and 2,593 misses (74.4% hit rate).
Cache promotions transferred 33.53 GB H2D in 3.679 s (9.11 GB/s). The profile
was not surveyed again. Compared with the previous 1.81 tok/s decode result,
the hybrid graph improved decode by about 38%; its short two-token prefill
measured below the earlier 1.51 tok/s result. The 10/20 tok/s target remains
unmet.

A `host_ram_gb=100` run was not admitted: its startup requirement is about
121.5 GB available, above the host's observed 114-117 GB idle range. The
34 GiB figure should be treated as total board use; 34 GiB of expert cache
alone cannot fit alongside the dense weights on these cards. The measured
37.66 GB device allocation includes the dense pool, expert caches, and other
runtime allocations. See the run log `/tmp/ling-3.1-flash-hybrid-full32-cache-split-90g.log`.

## Transfer and cache A/B follow-up (2026-10-19)

All runs below kept the raw two-token `Hi there` prompt and the existing
5,879-record usage profile unchanged; no full survey was run. They used the
90 GiB DDR pool, 3.0/14.4 GiB expert-cache budgets on the 3090/5060 Ti,
32 output tokens, and a 100 GiB `MemoryMax` with `MemorySwapMax=0`.

### VRAM pin-count sweep

| Pinned VRAM experts | Decode tok/s | Prefill tok/s | Cache hits | Cache-promotion H2D |
|---:|---:|---:|---:|---:|
| 1,024 | 2.49 | 1.29 | 75.5% | 29.60 GB |
| 2,048 | 2.49, repeat 2.54 | 1.32, repeat 1.29 | 74.4% | 33.53 GB |
| 3,072 | 2.70 | 1.32 | 72.5% | 35.36 GB |
| 4,096 | 2.40 | 1.35 | 71.1% | 36.18 GB |

The pin-count sweep did not produce a monotonic speed or hit-rate gain. The
single 3,072-pin result was fastest but also had the lowest hit rate among the
first three settings and the most H2D traffic; the 2,048-pin repeat stayed
close to its original result. Keep 2,048 pins for now; do not treat the single
3,072-pin measurement as a settled improvement.

### Decode asynchronous prewarm

Enabling `decode_async_prewarm` and `decode_async_prewarm_nonblocking` with a
three-expert batch measured 2.48 tok/s decode and 1.30 tok/s prefill. It
reported 74.3% cache hits and 33.12 GB H2D. The report also showed 3,171 of
8,207 prewarmed slots reclaimed before use (38.6%, 168.18 GB cumulative
traffic) and 3,173 duplicate synchronous fallback transfers (10.79 GB).
Leave decode prewarm disabled for this configuration.

### Persistent host-page pin budget

A 20 GiB `ram_pinned_budget_gb` attempt passed startup admission at 114.1 GB
available / 111.5 GB required, but its process exited with status 137 during
expert preload before a request result was produced. It ran with
`MemoryMax=100G` and `MemorySwapMax=0`; this is not a throughput result. Keep
the tested 15 GiB pin budget under the current host-memory headroom.

### Nsight request-window profile

A separate Nsight Systems capture of the 2,048-pin hybrid run was instrumented
and is not a throughput measurement. In the decode request window, it recorded
31.149 GB H2D (3.250 s device time), 0.419 GB D2H (0.095 s), 31,248 IQ1_S
matmul kernels (2.703 s), and 15,624 IQ2_XXS matmul kernels (0.767 s). It also
recorded 39,940 `cudaMemcpyAsync` calls (3.212 s cumulative API time), 32,091
`cudaStreamSynchronize` calls (2.455 s), and 2,610 `cudaGraphLaunch` calls
(0.178 s). These API durations overlap; they are not additive request-wall
time. The small graph-launch total makes per-expert compute, memory movement,
and synchronization better targets than graph-launch overhead alone. The
capture is `/tmp/ling-3.1-flash-hybrid-trace.nsys-rep`.

## Shared-input A/B admission (2026-10-19)

A follow-up code change reuses one hidden-state H2D input per GPU per
layer/token instead of copying the same input into every expert scratch
buffer. The change builds successfully and does not modify CUDA kernels.

The first 32-token A/B attempt used the same prompt/profile, the 90 GiB host
config, a 100 GiB cgroup cap, and `MemorySwapMax=0`. Engine startup admitted
114.1 GB against its 111.5 GB requirement, but the systemd scope ended with
`Result=oom-kill` during DDR preload, before a request was run. Its recorded
scope peak was 64.5 GB, below the child scope's 100 GiB `MemoryMax`; the
external cause is unknown. This produced no throughput result and does not
verify the shared-input code.

The A/B watcher now requires at least 116.5 GB available, no competing model
or build process, an idle GPU pair, and a free admission lock. The benchmark
itself remains capped at `MemoryMax=100G`, `MemorySwapMax=0`. The shared-input
change remains uncommitted until a clean 32-token request measures its effect.
