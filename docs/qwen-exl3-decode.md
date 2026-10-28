# Qwen EXL3 safetensors decode

EXL3 is experimental. The example profiles are maintainer-local files under the
ignored `configs/testing/` directory and are not shipped; the paths below are
placeholders for your own profiles. Setup, metadata requirements, and same-split
benchmarking are documented in [exllamav3-backend.md](exllamav3-backend.md).

To reuse ExLlamaV3's **complete optimized execution backend**, including its
GPU attention/cache/decode and AVX-512/VNNI CPU expert worker, add the
ExLlamaV3-specific keys described there to your profile.
The native EXL3 path described below remains available as the default.

Qwen3.8-Flash-Next EXL3 checkpoints can run through CPU or CUDA matrix dispatch
with `model_format = "exl3"` and `model_load_only = false`. The decoder reads
EXL3 `mul1` weights directly from safetensors, supports integer bitrates 1–8,
and streams the selected routed experts and PLE embedding rows from disk.
Dense weights are loaded lazily into host memory. Symlinked shards are accepted.

This checkpoint directory has no HF `config.json`, so the provided profile uses
`metadata_model_path` to read architecture and tokenizer metadata from a matching
Qwen GGUF. Its GGUF weight payloads are not used for EXL3 inference.

```bash
# Point these at your own local profiles; they are not in the repository.
EXL3_PROFILE=/path/to/qwen4exp_exl3_decode.toml
EXL3_GPU_PROFILE=/path/to/qwen4exp_exl3_gpu.toml

./build/zeromoe --config "$EXL3_PROFILE" --prompt "Hello" --predict 8
```

A load-only profile retains the catalog/payload smoke mode. For GPU
projections, use a CUDA build and the GPU profile:

```bash
./build/zeromoe --config "$EXL3_GPU_PROFILE" --prompt "Hello" --predict 8
```

`exl3_gpu = true` enables direct packed `mul1` CUDA matmul with 128-wide
Hadamard transforms, FP16 codebook decoding, FP32 accumulation and optional bias.
`exl3_gpu_device` selects the CUDA-visible device; `exl3_gpu_cache_gb` bounds
resident GPU weights. `exl3_gpu_expert_layer_begin = 26` puts routed experts
in layers 0–25 on the VNNI CPU path and layers 26–47 on the GPU, matching the
Qwen `-ncmoe 26` layer split. The profile divides those final 22 layers at 41:
layers 26–40 use `exl3_gpu_device`, and layers 41–47 use
`exl3_gpu_split_device`, which lets the 25.98 GiB routed-weight set span the
3090 and 5060 Ti. Their cache budgets are independently set by
`exl3_gpu_cache_gb` and `exl3_gpu_split_cache_gb`. Dense projections remain on
the primary device. The profiles use twelve OMP threads on this six-core,
12-thread CPU after the synthetic EXL3 thread sweep.
Dense weights are pinned; routed expert projections
use an eviction-safe LRU and avoid disk reads on cache hits. CPU routed experts
use AVX-512 VNNI `mul1` decoding and a bounded host cache controlled by
`exl3_cpu_expert_cache_gb` (24 GiB in the profile; set 0 to disable).
`exl3_cpu_expert_int8_vnni` enables VNNI byte-dot products with int8
activations and bypasses the per-weight FP16 codebook rounding step; the
synthetic matrix parity bound is 3% RMS. It was enabled in the local 26/22
profile used for these measurements; set it false for exact CPU matmuls. Unquantized F32
projections use cuBLAS. The default `exl3_gpu = false` retains CPU reference
dispatch; requesting GPU dispatch in a CPU-only build fails explicitly.

The 26/22 profile previously measured 3.31 tok/s over 127 decode intervals.
A later warm-cache, 128-token long-form run measured 5.50 tok/s over 127
intervals. With fused gate/up finishing and weighted down accumulation, the
same long-form prompt measured 6.50 tok/s over 127 intervals. That run reported
57.2 ms CPU matmul, 7.4 ms CPU weight loading, 41.1 ms GPU executor time,
8.4 ms GPU weight upload, and 39.7 ms remaining work per interval. Cache and
CPU timing varied between runs, so the change in total throughput cannot be
attributed solely to kernel fusion. An eight-token Hello comparison measured
3.29 tok/s with int8 VNNI and 3.58 tok/s with exact VNNI; both sampled the
same tokens. These runs are not directly comparable because cache warmth and
prompt differ. The 28+ tok/s target remains unmet.

A warmed comparison on the current standard 26/22 profile measured 6.39 and
6.41 tok/s with default OpenMP waiting, versus 7.10 tok/s with
`OMP_WAIT_POLICY=PASSIVE` and CUDA `auto`. All 128 token IDs and printed
selected-token logits matched. Use the passive launch environment for
ZeroMoE's native EXL3 hybrid profile; it does not apply to the delegated
ExLlamaV3 backend. [The waiting-policy report](exl3-wait-policy-2026-10-05.md)
contains raw logs and the cold-cache result excluded from the comparison.

The packed CPU tile-grouping change preserved all 128 generated token IDs in
a later comparison with the same long-form prompt. Concurrent compiler jobs
made that run unsuitable for throughput comparison. The added GPU timing
breakdown counted 761 standalone projection calls per decode interval;
standalone projections dominated GPU executor wall time over routed FFNs.

GPU-routed gate/up/down projections, SwiGLU and weighted accumulation stay on
the GPU through one FFN call and one activation transfer each way per MoE
layer. The input Hadamard and packed-trellis staging are inside each rotated
matmul; gate/up share one output Hadamard plus SwiGLU kernel, and the down
output Hadamard accumulates its routing-weighted result. This is five kernel
launches per expert. All selected weights are held during compute; evictions reuse compatible GPU buffers rather than
allocating/freeing each cold expert projection. PLE table locations and head
bias are indexed once, with persistent shard descriptors and a bounded
last-row cache per head; inference does not repeatedly scan the tensor catalog.

`exl3_gpu_hc = true` enables an experimental device chain for eligible F32
HyperConnection down/up/injection weights:
the low-rank SiLU activation, sigmoid gate and stream mixing stay on the GPU.
The block and output-head paths retain their respective division orders.
`hc_mixes` in the GPU statistics confirms that this path is selected.
The default is false. A local `qwen4exp_exl3_gpu_hc_experiment.toml` profile enables
the chain with the same 26/22 expert split. A 128-token checkpoint run used
the chain but first diverged from the previous greedy token sequence at
step 37; model-level accuracy assessment remains required before enabling
it in the standard profiles.

This is a hybrid decoder, not an all-GPU graph: attention/GDN state updates,
normalization and PLE row gathering remain on CPU. GDN recurrence uses the
existing vectorized delta-step implementation, parallel across independent
value heads. Dense projections still use synchronous activation transfers. Full-attention
layers retain the existing QSA indexer deferral. GPU reconstruction and matmul
parity tests cover integer bitrates 1–8, multiple rows, optional bias and cache
eviction. To run just those tests:

```bash
ZEROMOE_TEST_EXL3_ONLY=1 OMP_NUM_THREADS=4 ./build/zeromoe_tests
```

To benchmark the EXL3 CUDA kernels without opening a model, use synthetic
2560×640 expert matrices. The first command represents the 15 GPU expert
layers on device 0; the second represents seven layers on device 1:

```bash
make bench-exl3 BUILD_DIR=build ARGS="--device 0 --layers 15 --experts 10 --bits 3"
make bench-exl3 BUILD_DIR=build ARGS="--device 1 --layers 7 --experts 10 --bits 3"
```

The benchmark reports gate/down matmul time, a gate projection with activation
copies, separate/grouped projection dispatch, and routed FFN time with direct
launches and CUDA graph replay. Its
rates cover GPU kernels with resident
synthetic weights, not model loading, CPU layers, or full token generation.
On this machine at three bits, the 3090's 15-layer routed FFN took 3.658 ms
with direct launches and 2.963 ms with graph replay; the 5060 Ti's seven layers
took 1.663 ms and 1.617 ms. The direct kernel-only total was 5.321 ms across
22 GPU layers, while the full decoder still spends much longer in dense
projections, transfers, CPU experts, and the remaining forward path.

The CPU expert path has a separate synthetic benchmark using the same
2560×640 matrices and 26-layer/10-expert count:

```bash
make bench-exl3-cpu BUILD_DIR=build ARGS="--threads 6 --bits 3"
make bench-exl3-cpu BUILD_DIR=build ARGS="--threads 12 --bits 3"
```

HyperConnection dispatch has a model-free benchmark with the checkpoint's
width 2560, four streams and rank 320:

```bash
make bench-exl3-hc BUILD_DIR=build ARGS="--device 0 --blocks 97 --iterations 10"
```

Add `--omp-threads 12` to prime an OpenMP team before each GPU dispatch and
compare `OMP_WAIT_POLICY` and `ZEROMOE_CUDA_SCHEDULE` settings. The latter
is now honored by the EXL3 executor as well as the GGUF path.

The resident 97-block working set measured 16.910 ms for separate projection
calls versus 10.649 ms for the device chain on the 3090. On the 5060 Ti,
the same comparison measured 20.588 ms versus 13.097 ms. Synthetic hidden
outputs differed by 3.46e-8 relative RMS; injection outputs matched. These
are isolated HC timings. The checkpoint run measured 6.13 tok/s and changed
the generated sequence; the 28+ tok/s goal remains unmet.

`--predecoded` measures an experimental cached Q8 weight layout. In the
synthetic 26-layer run it reduced CPU matmul time from 42.4 ms to 35.3 ms at
12 threads, but converting all 780 matrices took 557 ms and the synthetic
parity bound is 4% RMS. The engine does not select this path because on-demand
conversion outweighs its per-token savings in the measured 128-token run.

`[EXL3-PERF]` reports first-to-last generated-token intervals separately from
prefill/startup, plus wall time spent in GPU executor calls, CPU matmuls, and
weight loads. These are host-wall measurements, not CUDA-event
kernel timings. GPU executor time is also split into standalone projection
calls and routed FFNs, so dense transfer/synchronization cost can be compared
with expert computation. `[EXL3-PLE]` reports row reads/cache hits; GPU cache statistics
include routed FFN calls and reused eviction buffers. The local profile used for these measurements uses
the matching 26-CPU/22-GPU routed-expert split, with GPU layers 26–40 on the
3090 and 41–47 on the 5060 Ti. The local profiles use 12 OMP threads. A synthetic
26-layer CPU expert benchmark measured 42.9 ms at 12 threads versus 55.0 ms
at six threads before the packed-tile grouping optimization. The int8 VNNI
path now shares activation/decode setup across four adjacent tiles for
bitrates 1–4 and two tiles for bitrates 5–8. Verify these changes with the full decoder before treating
that isolated gain as end-to-end throughput. The remaining attention/GDN and orchestration
path still runs through the CPU decoder rather than Bee's full GPU graph.
