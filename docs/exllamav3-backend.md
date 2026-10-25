# ExLlamaV3 execution backend

The `exllamav3` backend delegates the complete EXL3 inference pipeline to the
local ExLlamaV3 checkout: model construction, attention/GDN, HyperConnections,
KV and recurrent caches, routing, GPU experts, CPU experts, and generation.
ZeroMoE's native EXL3 kernels are not called on this path. The CLI replaces
itself with the configured Python process, using argv rather than a shell.

To enable the upstream backend, use an EXL3 TOML profile with
`model_format = "exl3"`, `exl3_backend = "exllamav3"`, the upstream source
directory, the isolated Python executable, and the desired CPU expert-layer
boundary. The checked-in 4bpw 26/22 and 2.05bpw 44/4 and 26/22 profiles are
working examples;
the `README.md` EXL3 section links to this guide.

The minimal backend-specific settings are:

```toml
model_format = "exl3"
exl3_backend = "exllamav3"
exllamav3_source_dir = "/home/ray/Projects/exllamav3"
exllamav3_python = "../.exl3-runtime/bin/python"
exllamav3_gpu_memory_gb = [22.0, 14.0]
exl3_gpu_expert_layer_begin = 26
omp_max_threads = 6
exllamav3_cuda_schedule = "blocking"
```

Set `model_path` to the safetensors checkpoint and supply metadata/tokenizer
paths when the model folder lacks the Hugging Face config files. Context,
generation length, prompt, and sampling options remain ordinary ZeroMoE
config entries. The profiles in `configs/` provide complete, tested examples.

On the Xeon W-2235 with the RTX 3090 and RTX 5060 Ti, six CPU-worker threads
are the measured default. Twelve threads contend with GPU submission and
nearly halve decode throughput.

The `exllamav3_cuda_schedule` setting selects the CUDA host wait mode
(`auto`, `spin`, `yield`, or `blocking`). It preserves unrelated CUDA device
flags and applies before loading the model. On this host, `blocking` is about
1.8% faster than `auto`.

The checkpoint includes an MTP draft head. Benchmark it with `--mtp`; the
report records its added load time and accepted draft-token count. Baseline
throughput does not use draft tokens.

`configs/qwen4exp_exllamav3_26_22.toml` requests the first 26 routed-expert
layers on ExLlamaV3's CPU worker and the remaining 22 on GPU. Actual per-layer
placement is checked after loading; ineligible offload silently falling back
to GPU is rejected. Dense/attention work remains on GPU. ExLlamaV3 autosplits
GPU modules under the configured per-device memory budgets, so the division
between the two GPUs need not match the native backend's 15/7 GPU allocation.

The CPU expert implementation is upstream `cpu/moe_mul1.cpp` (persistent
worker pool, runtime AVX2/AVX-512/VNNI dispatch). Upstream
`parallel/all_reduce_cpu_avx512.cpp` handles tensor-parallel communication,
not the expert matmuls. Whole-layer CPU offload requires layer-split mode,
not tensor-parallel mode.

## Environment

An isolated environment is used; no global Python installation or upstream
source modification is required:

```bash
uv venv --python 3.12 .exl3-runtime
uv pip install --python .exl3-runtime/bin/python torch --torch-backend cu130
uv pip install --python .exl3-runtime/bin/python \
    -r /home/ray/Projects/exllamav3/requirements.txt

export PYTHONPATH=/home/ray/Projects/exllamav3
export TORCH_EXTENSIONS_DIR="$PWD/.exl3-runtime/torch_extensions"
export TORCH_CUDA_ARCH_LIST='8.6;12.0'
export MAX_JOBS=4
export CUDAHOSTCXX=/usr/bin/g++-15
```

Use the installed host compiler and CUDA flavor appropriate to your machine.
The extension builds on first import. Keep the same extension directory and
architecture list for subsequent launches to reuse that build.
The supplied config sets this extension directory, architecture list, and
CUDA host compiler when they are not already overridden in the environment.

## Checkpoint metadata

The local checkpoint has all nine safetensors shards, the n-gram table, and
`tokenizer.json`, but no HF `config.json`. The upstream backend requires HF
metadata rather than the native decoder's GGUF metadata substitute.

The folder's `exl3_3bpw` name is not its tensor bitrate: independent header
inspection and `exl3_tool verify` find **75,277 4-bit matrices, two 5-bit
matrices, and 308 6-bit matrices**. For example, a routed expert trellis has
shape `[160, 40, 64]`, hence `64 / 16 = 4` bits/weight. The runner records
actual tensor bitrates in its benchmark JSON rather than inferring them from
the directory name. See `logs/exllamav3_runtime/verified_checkpoint.log` and
`logs/exllamav3_runtime/verified_rates.log`.

The validation metadata is retrieved from the official
[Qwen configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/21a7c3ca68484a9987d62fd51b78130d9b9a11f8/config.json)
at revision `21a7c3ca68484a9987d62fd51b78130d9b9a11f8`:

```bash
mkdir -p logs/exllamav3_runtime/metadata
for name in config.json tokenizer_config.json; do
    curl -fL "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/21a7c3ca68484a9987d62fd51b78130d9b9a11f8/$name" \
        -o "logs/exllamav3_runtime/metadata/$name"
done
```

The adapter creates a temporary directory of symlinks combining these small
metadata files with the existing checkpoint. It neither copies nor modifies
model weights, accepts symlinked source shards, rejects broken links, and
removes the view after worker shutdown. Relative paths in this backend's
config are interpreted relative to the TOML file, not the caller's directory.

## Run and benchmark

Validate configuration and model-view paths without importing CUDA:

```bash
python3 tools/run_exllamav3.py \
    --config configs/qwen4exp_exllamav3_26_22.toml --check
```

This does **not** prove tensor completeness, successful loading, or inference
correctness. Those require the upstream runtime to load and execute the model.

After rebuilding ZeroMoE, normal inference can use the same config entrypoint:

```bash
./build_ggmlcuda/zeromoe \
    --config configs/qwen4exp_exllamav3_26_22.toml --prompt "Hello" --ngen 8
```

For a fixed-length, greedy benchmark without early EOS stopping:

```bash
.exl3-runtime/bin/python tools/run_exllamav3.py \
    --config configs/qwen4exp_exllamav3_26_22.toml \
    --benchmark --warmup-tokens 8 \
    --output-json logs/exllamav3_runtime/benchmark_26_22.json
```

The benchmark uses upstream `Generator`/`Job` timings, separates loading and
prefill from decode, and reports `(generated_tokens - 1) / time_generate` to
match the native 128-token/127-interval comparison. The JSON includes token
IDs, completion text, source revision, PyTorch version, and observed layer
placement. Benchmark mode omits per-token console output. Normal inference
honors upstream EOS tokens. Both modes reject existing GPU compute processes
and hold ZeroMoE's shared host-admission lock throughout loading and inference.

## Measured results

On 2026-10-05, fixed 128-token runs used the same prompt and verified 26 CPU /
22 GPU layer placement:

| CPU worker threads | CUDA wait | Mode | Decode tok/s | Decode time |
| ---: | :--- | :--- | ---: | ---: |
| 12 | Auto | Fixed output | 15.99 | 7.944 s |
| 6 | Auto | Fixed output | 26.39 | 4.813 s |
| 6 | Blocking | Fixed output | 26.86 | 4.728 s |
| 6 | Blocking | ZeroMoE CLI, streamed output | 26.61 | 4.773 s |
| 7 | Blocking | Fixed output | 24.68 | 5.147 s |
| 5 | Auto | Fixed output | 25.98 | 4.888 s |
| 4 | Auto | Fixed output | 22.44 | 5.661 s |

The six-thread run is a 65% increase over twelve threads. Those two runs
produced identical prompt and generated token IDs across all 128 tokens. The
six-thread/blocking profile measures 26.61 tok/s through the ZeroMoE CLI.
Fixed-length output measures 26.86 tok/s. The 28 tok/s target remains unmet.
The upstream worker reports `avx512-vnni`; this Xeon
does not expose `avx512_vbmi`. Profiling shows most CPU expert time in gate/up
GEMV, followed by down GEMV. Setting upstream's `EXL3_MOE_CPU_SMALL_WORKERS=2`
reached 26.30 tok/s, so leave it unset. The optional MTP run accepted 68 draft
tokens and rejected 172 (28.3% acceptance), reducing decode to 16.51 tok/s.
See the individual logs and JSON under `logs/exllamav3_runtime/`.

The full-checkpoint runs verify loading, coherent output, and the requested
26/22 placement. CPU experts stay RAM-resident in the upstream worker rather
than using ZeroMoE's native on-demand CPU LRU. The 28 tok/s target remains
unmet; the measured peak is 26.86 tok/s. MTP is opt-in and measured slower.

## User-supplied 2.05bpw checkpoint

The requested `/mnt/usb-ssd/Qwen3.8_flash/exl3_2bpw` model passes
`exl3_tool verify`: five shards, 48 layers, 512 experts, payload smoke passed;
its actual trellis rates are 75,277 2bpw matrices, two 3bpw, and 472 4bpw.
This is a real 2.05bpw pack and is distinct from the earlier `exl3_3bpw`
directory, whose matrices were almost all 4bpw. The removable-drive model
folder contains no `tokenizer.json`, so the profiles symlink the matching
Qwen tokenizer from the existing 4bpw directory through
`exllamav3_tokenizer_path`; model weights still come from the 2.05bpw shards.
`exl3_tool verify` checked every tensor range and payload smoke test; its
record is `logs/exllamav3_runtime/exl3_2bpw_verify.log`.

Three verified-config profiles are available:

| Profile | CPU routed layers | GPU routed layers |
| --- | ---: | ---: |
| `configs/qwen4exp_exllamav3_2bpw_0_44_4.toml` | 44 | 4 |
| `configs/qwen4exp_exllamav3_2bpw_0_43_5.toml` | 43 | 5 |
| `configs/qwen4exp_exllamav3_2bpw_26_22.toml` | 26 | 22 |

For ExLlamaV3 whole-layer offload, these names mean layers `0..43` or `0..42`
run routed experts on the CPU worker, and the tail four or five layers run them
on GPU. Autosplit then places actual modules over the configured 3090/5060 Ti
budgets. I ran each profile's CPU-only `--check`; it verifies the checkpoint,
tokenizer and requested layer count, not CUDA loading or VRAM fit.

Verify the pack, then run the first profile on an otherwise idle host:

```bash
./build_ggmlcuda/exl3_tool verify /mnt/usb-ssd/Qwen3.8_flash/exl3_2bpw
./build_ggmlcuda/zeromoe --config configs/qwen4exp_exllamav3_2bpw_0_44_4.toml \
    --ngen 128
```

The fixed-length 0/44/4 benchmark has now loaded and generated 128 tokens:

| CPU routed layers | GPU routed layers | Load | Prefill | Decode | Placement/result |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 44 | 4 | 88.68 s | 1.496 s | 22.99 tok/s | Layers 0–43 on CPU; layers 44–47 on `cuda:0`; fit succeeded |

This run used six CPU worker threads, blocking CUDA waits, eight warmup tokens,
the same 42-token garden prompt, and fixed 128-token generation. The four GPU
expert layers fit on the RTX 3090; 15,306 MiB (about 14.95 GiB) remained free
after load. Although
the 5060 Ti was available, autosplit placed these four routed-expert layers on
`cuda:0`. Since 0/44/4 fit, the conditional 0/43/5 fallback was not run. The
22.99 tok/s result is below the requested 25 tok/s floor and is not directly
comparable to the 26/22 result above because the CPU/GPU expert-layer split is
different. Full output, placement, and timing are in
`logs/exllamav3_runtime/benchmark_2bpw_0_44_4.log` and
`logs/exllamav3_runtime/benchmark_2bpw_0_44_4.json`.

For a repeatable fixed-length run of this profile, use the same runner as the
26/22 benchmark:

```bash
.exl3-runtime/bin/python tools/run_exllamav3.py \
    --config configs/qwen4exp_exllamav3_2bpw_0_44_4.toml \
    --benchmark --warmup-tokens 8 \
    --output-json logs/exllamav3_runtime/benchmark_2bpw_0_44_4.json
```

The 2.05bpw pack was also tested at the target 26-CPU/22-GPU split, using the
same prompt, six CPU threads, blocking CUDA waits, eight warmup tokens, and
fixed 128-token output:

| Trial | CPU routed layers | GPU routed layers | Load | Prefill | Decode |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | 26 | 22 | 88.83 s | 0.977 s | 31.23 tok/s |
| 2 | 26 | 22 | 90.61 s | 0.970 s | 31.37 tok/s |
| Mean | 26 | 22 | — | — | 31.30 tok/s |

Both runs exceed 28 tok/s. Routed experts in layers 26–47 all landed on
`cuda:0` (RTX 3090); the 5060 Ti received no routed-expert layers. Prompt
token IDs match across runs; generated IDs match for the first 119 of 128
tokens and diverge in the final nine. Both completions remain coherent. Full
placement, token IDs, and timings are in
`logs/exllamav3_runtime/benchmark_2bpw_26_22.json` and
`logs/exllamav3_runtime/benchmark_2bpw_26_22_repeat.json`.

Repeat the fixed-length measurement with:

```bash
.exl3-runtime/bin/python tools/run_exllamav3.py \
    --config configs/qwen4exp_exllamav3_2bpw_26_22.toml \
    --benchmark --warmup-tokens 8 \
    --output-json logs/exllamav3_runtime/benchmark_2bpw_26_22_repeat.json
```

Q2_0 comparison work remains deferred until EXL3 performance tuning is
complete, as requested.

### Experimental GGML opaque EXL3 custom node

The repository also contains an isolated proof of the proposed GGML bridge:
`exl3_ggml_custom_bench` creates an `GGML_TYPE_I8` tensor containing one
packed EXL3 trellis/header/scale payload and passes it, together with an F32
activation tensor, through `ggml_map_custom2`. The custom callback validates
the opaque header, obtains the packed EXL3 components, launches
`launch_exl3_mul1_matmul`, and returns the F32 result. GGML owns the opaque
tensor allocation and graph node; the callback owns only the CUDA staging
buffers.

Build and run the isolated bridge benchmark with a CUDA+GGML build:

```bash
cmake --build build_ggmlcuda --target exl3_ggml_custom_bench -j4
./build_ggmlcuda/exl3_ggml_custom_bench 100
```

The corrected 2026-10-06 smoke benchmark measured 20.23 us through the GGML
graph and 19.50 us through the same direct callback path, or 0.73 us of GGML
graph/callback overhead for one synthetic projection. This validates the opaque
tensor and callback mechanics, not full-model EXL3 inference. GGML's current
custom operation is scheduled through the CPU backend; the callback enqueues
the EXL3 CUDA work and synchronizes it. A production Qwen EXL3 graph still
needs an EXL3 safetensors-to-GGML graph builder, persistent per-layer opaque
buffers, and full attention/KV-cache/logit integration. The GPU adapter below
supplies the CUDA-backend/custom-op handoff without modifying the external
GGML checkout. These benchmarks remain separate from the validated upstream
ExLlamaV3 path and do not replace it yet.

The GPU-resident adapter is `exl3_ggml_cuda_adapter_bench`. It wraps the CUDA
GGML backend without modifying the external GGML checkout, intercepts a custom
node, and invokes the EXL3 kernel using activation, opaque packed-weight, and
output device pointers directly. Its synthetic `2560 -> 640` projection
measured **13.79 us device time** (13.71 us host enqueue time) on the RTX 3090
on 2026-10-06; the adapter performs no per-node H2D/D2H copies.

For a stable graph, enable adapter-level CUDA graph capture with:

```bash
env -u CUDA_VISIBLE_DEVICES ./build_ggmlcuda/exl3_ggml_cuda_adapter_bench \
    100 --capture
```

The adapter captures and replays static GGML spans on either side of a dynamic
EXL3 callback, leaving route-dependent dispatch eager. Its replay signature
checks node identities, op/type, shape/strides, and node/source data addresses;
a mismatch clears the captured segments and recaptures. The adapter benchmark
now brackets its dynamic custom node with CUDA SiLU nodes to exercise both
captured spans. On 2026-10-08, the 10-iteration smoke confirmed **two captured
static segments around one eager custom callback** and measured **16.26 us GPU
time with segmented capture** versus **18.27 us eagerly** on the RTX 5060 Ti.
This tiny synthetic graph proves the segmented path executes, not a meaningful
full-Qwen speedup. The earlier **11.93 us** single-projection
capture result predates this segmented test graph and is not comparable. Call
`exl3_ggml_cuda_backend_reset_graph()` when replacing graph owners; weight
uploads and route-dependent buffer updates remain outside capture.

`exl3_ggml_cuda_real_bench` exercises the same bridge with a real matrix loaded
from `/data/sdb/models/qwen3.8-flash-next/exl3_3bpw`. The selected routed expert
gate projection is `2560 -> 640`, 4-bit, and 825,728 packed bytes. It measured
**12.88 us device time** (13.48 us enqueue-to-synchronize time), with a maximum
absolute GPU-vs-CPU EXL3 result error of `1.23e-7`. This validates real
safetensors loading, GPU-resident opaque storage, custom-node dispatch, and
single-matrix numerical parity; it is not full-model decode. Replaying the
same real node through adapter-level CUDA graph capture measured **11.06 us
device time** with the same `1.23e-7` parity error.

Run the real-loader check with:

```bash
cmake --build build_ggmlcuda --target exl3_ggml_cuda_real_bench -j4
env -u CUDA_VISIBLE_DEVICES ./build_ggmlcuda/exl3_ggml_cuda_real_bench \
    /data/sdb/models/qwen3.8-flash-next/exl3_3bpw 100 0 --capture
```

The fused `exl3_ggml_cuda_fused_ffn_bench` reduces gate/up, SwiGLU, and down
to one custom node with persistent CUDA intermediates. The captured synthetic
FFN measured **23.49 us device time** on the same GPU. This is an FFN kernel
fixture, not a Qwen layer because normalization, routing, residual, attention,
and KV-cache work are absent.

The reusable implementation is in `include/exl3_ggml_ffn.hpp` and
`src/exl3_ggml_ffn.cpp`. `Exl3GgmlFfnState` provides a single-expert fused
node, while `Exl3GgmlRoutedFfnState` accepts a compact top-k payload plus a
GPU-resident F32 route-weight tensor through `MAP_CUSTOM3`. A real two-expert
layer-0 check measured **61.05 us eagerly** and **52.03 us with adapter CUDA
graph capture**, with `1.75e-9` maximum absolute error against CPU gate/up,
SwiGLU, down, and weighted accumulation.
`exl3_ggml_build_routed_payload()` supplies the fixed-stride opaque layout used
by graph builders and validates all expert dimensions/bit metadata.

The routed state also has a dynamic route-table mode. GGML route outputs
(`I32` IDs and `F32` weights) feed `Exl3GgmlRoutePackState`, whose `MAP_CUSTOM3`
node packs them on the CUDA stream into an opaque `GGML_TYPE_I8` table storing
`[int32 route_ids[topk] | float route_weights[topk]]`. The captured EXL3 node
reads both arrays directly from device memory. A real two-expert test changed
the route from `[0,1]` to `[1,0]` after capture without rebuilding the graph,
measuring `1.75e-9` and `1.89e-9` maximum errors for the two replays. The
dynamic state now uses a validated uniform-bit fast path when all payload slots
share a bit width; mixed-bit payloads retain the eight-variant fallback.
Captured time for the real 4-bit test was **59.14 us**, versus **52.03 us** for
compact host-resolved routes.

The real routed check is reproducible with:

```bash
cmake --build build_ggmlcuda --target exl3_ggml_cuda_real_routed_ffn_bench -j4
env -u CUDA_VISIBLE_DEVICES ./build_ggmlcuda/exl3_ggml_cuda_real_routed_ffn_bench \
    /data/sdb/models/qwen3.8-flash-next/exl3_3bpw 100 0 --capture
```

Dynamic device-ID validation uses:

```bash
cmake --build build_ggmlcuda --target exl3_ggml_cuda_real_dynamic_routed_bench -j4
env -u CUDA_VISIBLE_DEVICES ./build_ggmlcuda/exl3_ggml_cuda_real_dynamic_routed_bench \
    /data/sdb/models/qwen3.8-flash-next/exl3_3bpw 20 0 --capture
```

`exl3_ggml_cuda_real_router_ffn_bench` extends this to the real layer-0 router:
GGML CUDA computes the F16 router matmul, softmax, top-k, selected-route
normalization, route-table packing, and dynamic EXL3 FFN. On the real pack it
selected experts `481,27`, matched CPU logits and normalized weights, and
measured **97.28 us eager** and **82.43 us captured** in the 2026-10-08
smoke, with `1.31e-9` maximum FFN error. This is a real routed layer-0 FFN
graph; attention, residual/HC, KV cache, and logits are still outside this
bridge benchmark.

`exl3_ggml_cuda_real_exl3_layer_bench` validates the next boundary using a real
full-attention EXL3 layer. Custom Q/K/V/O projections, RoPE, device KV-cache
write, GGML flash attention, Qwen HC mix/combine projections, residual adds,
and the routed EXL3 FFN execute in one captured graph. Layer 3
(`hidden=2560`, `heads=24`, `kv_heads=2`) measured **333.89 us** and produced
finite output.

The same graph accepts prefill rows (`S=2`), including row-strided route IDs,
weights, and batched EXL3 projections; the captured check measured **414.72 us**
with finite output.

The production Qwen full-graph builder has an opt-in selected-layer hook. It
replaces the selected GGML expert subtree with one cache-backed `MAP_CUSTOM3`
node. The node consumes device-resident activations and a packed route table.
The full-graph path resolves its packed EXL3 projections into ZeroMoE's
already allocated `GpuTopology` expert-cache slots (one slot per projection),
using negative cache IDs to keep them separate from GGUF expert identities.
It holds those slots until the graph stream completes; EXL3 eviction IDs are
not sent through the GGUF-to-DDR demotion callback. The DDR cache also uses the
already allocated selected-layer `GraphRamLayerArena`, divided into existing
per-expert-sized slots; each slot caches one packed projection. On a DDR miss,
the bridge reads Safetensors and fills an arena slot without allocating a new
host tensor buffer. Because this selected layer is replaced by EXL3, its GGUF
arena bytes are repurposed, and bridge construction fails closed rather than
falling back to those overwritten weights. Slot size/device mismatch also
fails closed. The GGML bridge is C++/CUDA and does not call Python or modify
the external GGML checkout.

This topology-backed full-graph path is separate from `model_format="exl3"`'s
native `CPUEngine` route. The latter still uses the `Exl3GpuExecutor` weight
LRU and `Exl3CpuMatrixCache`; the cache-graph benchmark below also tests that
executor-owned cache path, not the full-graph topology adapter.

```bash
export ZEROMOE_EXL3_GGML_BRIDGE_MODEL_PATH=/data/sdb/models/qwen3.8-flash-next/exl3_3bpw
export ZEROMOE_EXL3_GGML_BRIDGE_LAYER=3
```

The full-graph bridge requires a complete preloaded DDR arena for each selected
layer and an existing per-layer GPU slot region large enough for at least one
top-k row. It reuses those arenas and fails closed when either tier is
unavailable; no all-experts EXL3 payload is allocated.

The same hook can be configured directly in TOML:

```toml
exl3_ggml_bridge_model_path = "/data/sdb/models/qwen3.8-flash-next/exl3_3bpw"
exl3_ggml_bridge_layer = 3
exl3_gpu_cache_gb = 8.0
# Optional, experimental: capture static GGML spans around dynamic EXL3 FFNs.
# Route-dependent cache lookup remains eager. Defaults to false.
exl3_ggml_graph_capture = false
# More experimental: cache CUDA graphs for repeated exact routed-FFN pointer
# signatures. Requires segmented capture and defaults to false.
exl3_ggml_routed_graph_capture = false
# Topology-backed full graph uses its existing DDR arena; the CPU matrix-cache
# budget is for model_format="exl3" native inference.
exl3_cpu_expert_cache_gb = 0.0
# Or replace the single layer with an explicit list:
# exl3_ggml_bridge_layers = [3, 7, 11]
# All routed FFNs can be selected with [0, 1, ..., 47] for this 48-layer Qwen.
```

The bridge validates `exl3_gpu_cache_gb` in `(0, 1024]` and initializes an
executor on the selected core GPU for persistent scratch and dispatch. Selected
layers share a process-wide Safetensors tensor index, avoiding a full shard
header rescan per layer. The custom node keeps its activation, route weights,
and output on CUDA. GGML's I32 route-ID and F32 route-weight tensors feed the
custom node directly; it no longer allocates an opaque route-table tensor or
launches a route-pack kernel. For device-produced routes, it copies only expert
IDs into a persistent pinned host buffer before resolving cache slots; route
weights are copied back only when CPU parity checking is enabled. This
synchronization is still a decode bottleneck. The focused EXL3 suite passes the
pinned-buffer D2H staging test and host-route setup on the RTX 5060 Ti; this is
not an end-to-end timing. The FFN kernels are then enqueued in
row batches sized to the unpinned per-layer slot region
(`slot_count / (3 * topk)`). Intermediate batches synchronize before reusing
those slots; the final batch remains asynchronous on the GGML CUDA stream, with
completion events retaining cache-slot holds until the work finishes. The next
device-route readback synchronizes that stream and reaps completed holds before
resolving new slots. This avoids a second end-of-node wait while preserving
safe slot reuse. Route readback keeps the routed FFN callback itself outside
CUDA graph capture. The adapter captures static GGML spans before and after
that callback; it checks the graph's
node signature before replay so changed graphs do not reuse stale segments.
Set `ZEROMOE_EXL3_GGML_ROUTE_TIMING=1` to log per-layer route-readback count,
bytes, cumulative host wait, and average synchronization time at teardown.
The external path now computes the gate and up rotated matmuls in one CUDA
kernel, staging both trellises and their input transforms together. This
reduces the per-route sequence from five launches to four; SwiGLU/output
transforms and weighted down projection remain separate. The updated kernel
compiles, and `zeromoe_tests` includes a 3-bit gate / 5-bit up parity case.
The external-pointer routed FFN and mixed-bit gate/up tests both pass on GPUs 0
and 1 after this change. A capture-enabled full-graph smoke is measured below;
it is not a stable performance comparison. A real full-graph attempt found
router weights were rank-3
`[1, topk, rows]`; the builder now makes them contiguous and reshapes to
`[topk, rows]`. Requested bridge construction now fails instead of silently
substituting GGUF expert weights.

The 2026-10-08 single-layer topology-backed smoke completed a 13-token prefill
and four generated tokens with EXL3 replacing layer 3 of the IQ1_M GGUF full
graph. It referenced the existing 904,396,800-byte DDR arena (512
1,766,400-byte slots) and GPU expert pool; shutdown reported 46 of 2,193 GPU
slots filled. Decode measured **3.79 tok/s**.

Set `ZEROMOE_EXL3_GGML_VERIFY_CPU_LAYER` to a layer number (for example `3`)
to compare one layer, or to `all` to compare every bridged layer, against the
CPU EXL3 reference. With `ZEROMOE_EXL3_GGML_VERIFY_CPU_LAYER=3`, the same
full-graph custom node compared topology-slot GPU outputs against the CPU EXL3
reference on all 13 prefill rows. Maximum absolute error was **7.83e-8**. This
verifies the real Safetensors -> existing DDR slot -> existing GPU slot ->
custom-op path for that layer, not just synthetic external-pointer dispatch.

A follow-up enabled the hook for all 48 routed FFN layers. The 13-token prefill
and four-token decode completed without bridge fallback or custom-op errors.
Each layer reused its existing 512-slot DDR arena (1.64–1.77 MB per slot) and
GPU cache region; the GPU cache ended at 6,635 of 6,749 slots filled (11.30
GiB). This mixed run retains IQ1_M GGUF dense/attention weights and replaces
the routed FFNs with EXL3, so it is not all-EXL3 logit parity. It measured
**2.88 prefill tok/s** and **1.72 decode tok/s** over four tokens. The row batch
was four at 140–141 available projection slots per layer; synchronization
between batches is a correctness/capacity tradeoff and a clear performance
optimization target, not an ExLlamaV3 comparison.

After gate/up fusion and segmented capture wiring, a new three-token prompt
with `exl3_ggml_graph_capture = true` completed prefill and three generated
tokens with the bridge selected for all 48 routed FFNs. The mixed GGUF/EXL3
run measured **1.56 prefill tok/s** and **1.55 decode tok/s** (0.78 overall
tok/s including prefill). This short smoke confirms end-to-end execution with
capture enabled; its duration and prompt are too small to establish a
performance gain or compare fairly with the earlier four-token run. A separate
adapter test confirms two static segments execute around an eager custom
callback.

The current direct-route path passed an all-layer run with
`ZEROMOE_EXL3_GGML_VERIFY_CPU_LAYER=all`: 144 custom FFN checks across all 48
layers, with maximum absolute error `2.75671e-7`. The three-token prompt and
three generated tokens completed with capture enabled. The parity callback now
waits for the asynchronous final FFN batch before reading its output. CPU
reference matmuls and device readbacks make this run diagnostic, not a
throughput measurement; it checks routed FFN outputs, not end-to-end logits
against an all-EXL3 model.

A non-parity run exercised the same direct I32/F32 route tensors, pinned ID
readback, fused gate/up kernel, and `exl3_ggml_graph_capture = true` across all
48 routed FFNs on the RTX 3090. It completed a three-token prompt and generated
three tokens; logs show 144 routed-FFN calls and a 49-segment graph around 48
eager routed callbacks. Measured rates were **1.55 prefill tok/s** and **3.27
decode tok/s** (1.05 overall tok/s). Executor counters report 7,200 projection
matmuls and **67.36 ms GPU FFN time** across those 144 calls, with zero executor
cache uploads because production weights come from topology slots. Route-sync
timings are around **0.3 ms per layer callback**, but include waiting for prior
router/graph work on the CUDA stream, not just the 40-byte ID copy. This short
mixed-format smoke is not a stable A/B performance claim or full-model
logit-parity check.

The all-layer smoke used `exl3_ggml_bridge_layers = [0, ..., 47]`,
`ram_preload_all_experts = true`, and
`ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=0` so every layer retained its DDR
arena for EXL3 slots. Reproduce with a Qwen4Exp GGUF metadata/model config,
the EXL3 Safetensors path, and a short prompt; the rest of the model's dense
and attention weights remain GGUF in this mixed-format integration test.

After fixing routed output initialization, an all-48-layer topology-backed
parity run completed a 13-token prefill and one decode token on the RTX 3090.
The callback compared 192 routed FFN outputs across prefill row batches and
decode, with maximum absolute error `4.17233e-7`; production slot-backed
dispatch recorded zero executor cache uploads. This confirms the all-layer
Safetensors -> existing DDR/GPU expert slots -> GGML custom-op path after the
output-clear fix. The parity run is deliberately diagnostic: CPU references
and readbacks reduced prefill to `0.60 tok/s`. The one-token decode interval
was `3.85 tok/s`, too short for a stable performance claim. It verifies routed
FFN outputs, not end-to-end logits against an all-EXL3 model.

For an end-to-end same-history comparison, set
`ZEROMOE_EXL3_GGML_VERIFY_CPU_LAYER=all` and
`ZEROMOE_EXL3_GGML_CPU_REFERENCE_OUTPUT=1` on the reference run. Each bridge
callback still checks GPU FFN parity, then substitutes that callback's CPU
reference output into the same full graph. Dump logits in both the ordinary
GPU-output run and this CPU-reference-output run with
`ZEROMOE_DECODE_LOGITS_DUMP`; use `ZEROMOE_GENERATED_TOKEN_FILE` on a
deterministic warmup and `ZEROMOE_TEACHER_FORCE_TOKEN_FILE` plus
`ZEROMOE_TEACHER_FORCE_SKIP_SAMPLING=1` for both compared runs so every
decode step sees the same token history. Compare the dumps with
`python3 bench/compare_decode_logits.py`. The CPU-reference substitution has
passed the focused routed-FFN benchmark, but a full-model logits comparison
has not yet been run. The end-to-end sequence is automated by
`python3 bench/exl3_ggml_logits_parity.py --out-dir /tmp/exl3-logit-parity`;
it fails fast if another `zeromoe` process is running or less than 110 GiB
of RAM is available, and verifies `nvidia-smi` is responsive before each run;
it then writes both logits and a comparison report there.
The CPU-reference-output hook was exercised on the RTX 5060 Ti with a single
external routed FFN (`rows=1`, `topk=2`); its result matched the CPU output
exactly (`max_abs_error=0`). The measured `30.5 ms` callback includes CPU
reference recomputation and is a correctness check, not a speed result.

The earlier full-graph smoke runs below exercised a fixed opaque layer payload;
they are historical evidence and do not verify this cache-backed implementation.
The current cache-backed benchmark loads selected Safetensors experts lazily,
alternates between two routed-expert sets across replays, compares each FFN
output against its CPU EXL3 reference, and checks the upload count:

```bash
cmake --build build_ggmlcuda --target exl3_ggml_cuda_cache_graph_bench -j4
env -u CUDA_VISIBLE_DEVICES ./build_ggmlcuda/exl3_ggml_cuda_cache_graph_bench \
    /data/sdb/models/qwen3.8-flash-next/exl3_3bpw 8 0 2
```

The 2026-10-08 direct-device-route rerun (`rows=1`, `topk=2`, two iterations,
RTX 5060 Ti) passed CPU parity (`max_abs_error=1.28057e-9`) at `avg_us=91.78`.
It performed four pinned route-ID readbacks (32 bytes total), averaging
**7.07 us per synchronization**, with 12 GPU uploads and 12 GPU-cache hits.
Route weights stayed on CUDA. This focused benchmark exercises the executor's
weight cache rather than the production topology-slot resolver, so it measures
route tensor staging and custom-op execution, not full topology-backed decode.
It also does not establish DDR-cache-hit performance or full-model EXL3 logit
parity. Do not use the historical fixed-payload throughput numbers as the
performance result for this implementation.

The same bench's `--external` mode drives `routed_ffn_device_external` with
Safetensors matrices staged to external CUDA pointers, matching the bridge's
non-owning-weight dispatch contract. On the RTX 5060 Ti (`rows=1`, Qwen
`topk=10`, two route sets, two measured iterations after warmup), it passed CPU
parity inside the custom callback for both warmup route sets (`max_abs_error`
`7.79e-10` and `7.57e-10`). A short follow-up measured `avg_us=348.91` wall /
`gpu_us=349.95`, with two pinned route-ID readbacks (80 bytes total) at
`avg_us=6.35` and `0.212 ms` total FFN enqueue time for the two measured
callbacks; executor cache uploads remained zero. These short runs varied
(`gpu_us` 277–350), so treat them as correctness and callback-cost checks, not
a stable performance result. The test allocates its own fixture pointers and
does not exercise the production `GpuTopology` resolver or full-model speed.

The same mode now runs its warmup route sets through the custom op's internal
CPU parity checker before timing. With `topk=2`, both route sets passed
(`max_abs_error=1.35e-9` and `1.28e-9`); route-readback synchronization averaged
`6.68 us`. This specifically verifies pinned direct-ID readback and the
parity-only output wait after asynchronous FFN dispatch.

A follow-up on 2026-10-08 found and fixed an external-route output initialization
bug: `cudaMemsetAsync` had its byte-count and fill-value arguments reversed,
leaving the accumulation buffer uncleared. After correcting the argument order,
the RTX 5060 Ti external-view benchmark passed both `topk=2` route sets
(`max_abs_error=1.35333e-9` and `1.28057e-9`, `avg_us=106.67`,
`gpu_us=109.52`). With `topk=1` and a repeated route signature, the opt-in
per-FFN CUDA graph captured once and replayed three times with parity at
`2.00089e-9` (`avg_us=82.17`, `gpu_us=85.42`). With `topk=2` and changing
route signatures, the cached graph correctly fell back eagerly for the two
unseen signatures; both still passed parity (`graph_captures=1`,
`graph_replays=1`, `graph_fallbacks=2`). These are small single-FFN correctness
checks, not full-model throughput or topology-backed decode validation.
The EXL3-only test suite also repeats an external routed FFN into the same
nonzero output buffer and checks that the second result does not accumulate the
first call's output.

```bash
CUDA_VISIBLE_DEVICES=1 ZEROMOE_EXL3_GGML_ROUTE_TIMING=1 \
  ./build_ggmlcuda/exl3_ggml_cuda_cache_graph_bench \
  /data/sdb/models/qwen3.8-flash-next/exl3_3bpw 2 0 1 --external 10
```

An end-to-end smoke used `model_format="exl3"`, the EXL3 Safetensors folder,
and the matching GGUF metadata, with GPU 0 enabled and one generated token.
Prefill completed for 13 prompt rows; all 48 transformer layers ran, including
the GGML custom-op routed FFN path on GPU layers 26–47 (22 routed FFNs). The
run completed in 63.49 s and reported 7.67 GiB resident in the EXL3 GPU LRU.
This proves the native EXL3 decoder's GGML custom-op path is reachable, but it
is only a smoke: one generated token is not a decode-speed measurement, and
there was no independent full-logit/token parity check. This path still uses
`Exl3GpuExecutor`/`Exl3CpuMatrixCache`; it does not exercise the topology-backed
full-graph hook above.

A repeated four-token run with expert layers 0–25 on CPU and 26–47 on GPU
completed three decode intervals at **1.56 tok/s** (`prefill_s=49.86`, total
`52.10 s`). The GPU reported 88 routed FFNs total (22 for prefill plus 22 per
decode interval),
`gpu_routed_ffn_ms_per_interval=6.83`, and
`cpu_matmul_ms_per_interval=53.52`. This confirms repeated execution through
the custom-op path but also shows the current mixed CPU/GPU route is not
performant; it is not a comparison against ExLlamaV3 or a fully GPU-resident
MoE placement.

```bash
cmake --build build_ggmlcuda --target exl3_ggml_cuda_real_exl3_layer_bench -j4
env -u CUDA_VISIBLE_DEVICES ./build_ggmlcuda/exl3_ggml_cuda_real_exl3_layer_bench \
    /data/sdb/models/qwen3.8-flash-next/exl3_3bpw 5 0 --capture
```

Run it with:

```bash
cmake --build build_ggmlcuda --target exl3_ggml_cuda_real_router_ffn_bench -j4
env -u CUDA_VISIBLE_DEVICES ./build_ggmlcuda/exl3_ggml_cuda_real_router_ffn_bench \
    /data/sdb/models/qwen3.8-flash-next/exl3_3bpw 10 0 --capture
```

The follow-up `exl3_ggml_cuda_moe_bench` exercises 48 chained routed-FFN nodes
with top-k=10, keeping opaque payloads and intermediates on CUDA. On
2026-10-06 it measured **11.00 ms device time eagerly** (90.9 synthetic
MoE-only steps/s) and **9.30 ms with CUDA graph capture** (107.5 synthetic
MoE-only steps/s). The synthetic chain intentionally omits Qwen normalization,
residual, attention, router, and KV-cache operations, and its random weights
overflow after repeated layers; `non_finite` is therefore expected. These
figures measure bridge dispatch and kernel work, not end-to-end model decode.
Full Qwen EXL3 graph construction, logit parity, and decode benchmarking remain
outstanding.

## Tests

```bash
python3 -m unittest discover -s tests -p test_exllamav3_backend.py -v
make test-components ARGS="exllamav3_backend_launch"
```

These cover configuration, symlinked views, cleanup, layer-placement checks,
prompt formatting, GPU-process rejection, and the no-shell C++ launch adapter.
The latest Python run on 2026-10-05 passed all 11 tests.
The runtime evidence above separately validates loading, actual routed-layer
placement, and fixed-length decode for the 4bpw 26/22 and 2.05bpw 44/4 and
26/22 profiles. Unit tests and config checks do not replace full-checkpoint
quality validation.

## Performance investigation

The local directory is named `exl3_3bpw`, but its trellis headers show 75,277
4bpw matrices, two 5bpw matrices, and 308 6bpw matrices. The official
[Turboderp EXL3 repository](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.05bpw_h5_ng5)
has a real 3.05bpw Qwen3.8 pack. The 4bpw checkpoint therefore sends about a
third more packed expert data through the RAM/CPU/GPU pipeline than a 3.05bpw
pack would, before scale and padding overhead.

The [ExLlamaV3 release benchmarks](https://github.com/turboderp-org/exllamav3/releases/tag/v1.5.0)
report 62.77 tok/s on Qwen3.8 at 3.05bpw using a 3090 Ti + 3090, an 8-core
AVX2 CPU, and 20% CPU offload with a pinned arena and the n-gram table in RAM.
That hardware, quant, offload fraction, and memory policy differ substantially
from this machine's 4bpw pack, six-core W-2235, 3090 + 5060 Ti, and 26 CPU /
22 GPU layer split; the number is context, not an apples-to-apples target.

I could not find an official ExLlamaV3/TabbyAPI result backing the quoted
"up to 2.5x faster than llama.cpp" prompt-processing figure. The official
release table instead compares ExLlamaV3 versions: its Qwen3.8 3.05bpw Pro
6000 row shows prefill rising from 5,788 to 7,327 tok/s (+27%), not 2.5x
versus llama.cpp. The separate 80%-CPU-offload row is on different hardware
and also compares release versions. These are prompt-prefill figures; this
profile used a 256-token context and a 42-token prompt, so our 26.61 tok/s
streamed CLI result measures decode and says nothing about 2,048-token
prompt processing. A valid 2,048-token comparison needs a matched prompt,
model quant, GPU set, offload split, prefill chunk, and timer boundary across
both backends.

VNNI is already running here, and the upstream [CPU expert tuning guide](https://github.com/turboderp-org/exllamav3/blob/master/doc/env_vars.md)
reports that swizzling improves its cold-expert GEMV bandwidth (+40% in its
7960X test). This runner leaves the upstream swizzle and CPU affinity defaults
enabled. The six-thread profile's own phase timings show gate/up GEMV as the
largest CPU phase, followed by down GEMV. Intel rates this W-2235 at four
DDR4-2933 channels and 93.85 GB/s theoretical memory bandwidth, but we have
not collected hardware DRAM counters to determine how much of that bandwidth
these EXL3 GEMVs sustain. The CPU worker also lacks AVX-512 VBMI, so this host
uses ExLlamaV3's AVX-512 VNNI tier.

The same upstream guide leaves the n-gram table streamed from disk by default
for SSD-class storage. This checkpoint's symlink resolves to a SATA SSD, so
loading the entire tens-of-gigabytes table into RAM is not yet justified as a
decode fix. The local MTP run accepted only 68 of 240 proposed draft tokens
(28.3%) and decoded more slowly, so MTP stays opt-in.

### 4bpw per-layer expert split (recommended 4bpw default)

Whole-layer 26/22 on the 4bpw pack peaks at 26.86 tok/s. Upstream's per-layer
expert split (`exllamav3_cpu_expert_split = 278`) runs the tail 278 routed
experts of every layer on CPU and 234 on GPU, a footprint roughly equal to
26/22, and overlaps CPU and GPU expert compute within each layer. It is
mutually exclusive with `exl3_gpu_expert_layer_begin`.

Upstream swaps hot and cold experts dynamically by default, which made decode
vary between 28.48 and 26.86 tok/s. Setting `exllamav3_cpu_expert_swap = false`
(the runner exports `EXL3_MOE_CPU_SWAP=0`) pins placement statically:

| Trial | Decode |
| :--- | ---: |
| 1 | 28.34 tok/s |
| 2 | 28.07 tok/s |
| 3 | 28.18 tok/s |
| Mean | ~28.20 tok/s |

All three trials clear 28 tok/s. This split with static placement is the
recommended 4bpw profile: `configs/qwen4exp_exllamav3_4bpw_cpu_split_278.toml`.
The key defaults to `true` (upstream dynamic swapping) when omitted.

### 2.05bpw expert-split sweep (VRAM-first notation)

Notation: **GPU / CPU routed experts per layer**, VRAM first. Each layer has 512
experts, 24,576 in the model. "Split layers / CPU-only layers" counts layers.
Earlier docs and configs wrote `0/44/4` and `44/4` for whole-layer offload, where
the first number was the CPU side; those numbers are not comparable to the
tables here. Pack: `/mnt/usb-ssd/Qwen3.8_flash/exl3_2bpw`. Settings: six CPU
threads, blocking CUDA waits, eight warmup tokens, 42-token prompt, fixed
128-token output, `context_size = 256`, static placement. Each row is one trial.

Expert size, read from the shard headers: about 1.19 MiB per expert in layers
1-47 (2bpw) and 2.38 MiB in layer 0 (4bpw). One extra GPU expert per layer
across all 48 layers costs about 58 MiB of VRAM; measured steps matched
(about 56 MiB).

Layer mix at 256 GPU / 256 CPU experts on split layers (needs
`exllamav3_cpu_mixed = true` unless all 48 layers are split):

| Split layers / CPU-only layers | Experts in VRAM | Free on 3090 | Decode |
| :--- | ---: | ---: | ---: |
| 44 / 4 | 11,264 | 4,534 MiB | 34.25 tok/s |
| 45 / 3 | 11,520 | 4,154 MiB | 34.75 tok/s |
| 46 / 2 | 11,776 | 3,894 MiB | 35.18 tok/s |
| 47 / 1 | 12,032 | 3,634 MiB | 35.86 tok/s |
| 48 / 0 | 12,288 | 3,252 MiB | 36.55 tok/s |

Shifting more experts to the GPU in all 48 layers (3090 only):

| GPU / CPU per layer | Experts in VRAM | Budget | Free on 3090 | Decode |
| :--- | ---: | :--- | ---: | ---: |
| 256 / 256 | 12,288 | 22 GiB | 3,252 MiB | 36.55 tok/s |
| 272 / 240 | 13,056 | 22 GiB | 2,372 MiB | 38.05 tok/s |
| 288 / 224 | 13,824 | 22 GiB | 1,472 MiB | 39.45 tok/s |
| **296 / 216** | **14,208** | **23 GiB** | **1,092 MiB** | **40.42 tok/s** |

Findings:

- Decode rises monotonically with GPU-resident experts; 296/216 is +10.6% over
  256/256. No row is repeated yet, so differences under about 0.5 tok/s are
  within the noise seen elsewhere.
- **304 / 208 is not a valid result.** With a 22 GiB budget and the 5060 Ti
  available, autosplit hit CUDA out of memory on layer 46 and moved layers 46-47
  to the 5060 Ti (40.25 tok/s, prefill 1.68 s). Cross-device spill is a
  different configuration. For 3090-only tests set the second budget to 0 so a
  spill fails loudly, and confirm every placement entry is `cuda:0`.
- 296/216 needs `exllamav3_gpu_memory_gb = [23.0, 0.0]`. At 22.0 it failed with
  OOM on `lm_head` (22.02 GiB allocated against the budget; the card has
  23.56 GiB). Only about 0.5 GiB of device headroom remains, so a longer
  context or larger chunk may not fit. This was not tested.
- The 5060 Ti held no routed experts in any 3090-only run.
- In all rows output was coherent and shared the same opening text; tails
  differed after about 90 characters, as in the earlier variance note.

How to run a split:

```bash
# 1. GPUs must be idle (the runner refuses otherwise)
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
pgrep -af 'llama|ollama|vllm|zeromoe'
# 2. Validate the config without CUDA
.exl3-runtime/bin/python tools/run_exllamav3.py --config <cfg>.toml --check
# 3. Benchmark; the JSON records placement, footprint and token ids
.exl3-runtime/bin/python tools/run_exllamav3.py --config <cfg>.toml \
    --benchmark --warmup-tokens 8 --output-json logs/exllamav3_runtime/<name>.json
```

Config keys (root level, before any table header):

```toml
exl3_gpu_expert_layer_begin = 0    # CPU-only layers (whole-layer offload), taken from layer 0
exllamav3_cpu_expert_split = 216   # CPU experts per split layer; GPU gets 512 - this
exllamav3_cpu_mixed = true         # only when both keys above are nonzero
exllamav3_cpu_expert_swap = false  # static placement (EXL3_MOE_CPU_SWAP=0)
exllamav3_gpu_memory_gb = [23.0, 0.0]  # [3090, 5060 Ti]; 0 keeps the test 3090-only
```

`exllamav3_cpu_mixed` relies on an uncommitted patch in the upstream checkout
(`moe_cpu_mixed` / `EXL3_MOE_CPU_MIXED` in `exllamav3/model/config.py` and
`modules/block_sparse_mlp_cpu.py`) that lifts the offload-versus-split
exclusion; the first N eligible layers go wholly to the CPU and later layers
get the split. Configs are `configs/qwen4exp_exllamav3_2bpw_*.toml`
(`mixed_0_44_4`, `mixed_0_45_3`, `mixed_0_46_2`, `mixed_0_47_1`,
`split_256_48_0`, `split_240_48_0`, `split_224_48_0`, `split_208_48_0`,
`split_216_48_0_3090only`); logs and JSON are in `logs/exllamav3_runtime/`.

The 2.05bpw 26/22 profile reaches a repeatable 31.30 tok/s mean and exceeds
the 28 tok/s objective. On the 4bpw pack, the post-graph blocking baseline was
26.67 tok/s; CUDA `spin` reached 26.26 tok/s, disabling CPU pinning reached
22.27 tok/s, and `EXL3_MOE_CPU_SMALL_WORKERS=1` reached 24.13 tok/s. The
two-worker small-job cap reached 26.30 tok/s. A diagnostic CPU profile
measured steady gate/up GEMV at 581 us/job and down GEMV at 290 us/job; it is
instrumented and its throughput is not a baseline. The historical 4bpw peak
remains 26.86 tok/s for whole-layer placement, 4.1% short of 28; the 4bpw
pack clears 28 only with the static per-layer expert split above. CPU expert gate/up GEMV remains the primary
optimization target for the 4bpw pack. Resume Q2_0 comparisons after the EXL3
performance work is complete.

## Upstream ExLlamaV3 kernels for the in-graph EXL3 path (no libtorch)

All EXL3 matmuls in the native path (`exl3_gpu`, the GGML custom-op bridge) run
on ExLlamaV3's own CUDA kernels; ZeroMoE no longer carries its own EXL3
matmul kernels. Build with the upstream extension sources:

```bash
cmake -S . -B build_ggmlcuda \
    -DZEROMOE_EXLLAMAV3_SRC=/home/ray/Projects/exllamav3/exllamav3/exllamav3_ext
```

- Upstream `exl3_gemm` / `exl3_gemv` / `exl3_moe_coop` are compiled unmodified
  as `zeromoe_exl3_upstream`. `third_party/exl3_torch_stub` stands in for the
  few ATen/c10 symbols their launchers use; libtorch is not required.
- `src/exl3_upstream_ops.cu` implements the `launch_exl3_mul1_*` API on
  `exl3_gemm` and the external routed FFN on `exl3_moe_coop` (two launches per
  layer for up to 8 rows / 256 route slots). Larger or mixed-bit-width batches,
  and biased matrices, use one upstream `exl3_gemm` sequence per expert.
- Without `ZEROMOE_EXLLAMAV3_SRC` the matmul entry points return
  `cudaErrorNotSupported` (`cuda/exl3_aux.cu`); the GPU EXL3 path is then
  unavailable. `cuda/exl3_aux.cu` also holds the hyper-connection, SwiGLU and
  accumulate helper kernels.
- `ZEROMOE_EXL3_MOE_COOP=0` disables the fused routed-FFN path (A/B).
  Upstream also defaults to an int8-activation GEMV for one row and K <= 5 on
  Ampere; `EXL3_INT8_GEMV=0` disables it.
- Upstream kernels take fp16 activations (about 5e-4 relative RMS against fp32;
  ~7e-3 on the int8 GEMV path), so tolerances against the fp32 CPU reference
  are wider than they were: 2e-2 of the output peak per routed FFN in the
  bridge parity check, 2e-2 relative RMS in the in-tree matmul test.
- `tests/exl3_reference_matmul.cu` keeps ZeroMoE's original fp32 kernels as a
  test-only independent reference. `tests/exl3_upstream` (standalone CMake
  project) builds `exl3_upstream_parity`, which compares it with upstream
  `exl3_gemm` on random matrices without a model.
- Removed with the legacy kernels: the payload-based dynamic route-table state
  (`Exl3GgmlRoutedFfnState::initialize_dynamic`, `Exl3GgmlRoutePackState`) and
  the benches that only exercised it (`exl3_ggml_cuda_real_dynamic_routed_bench`,
  `exl3_ggml_cuda_real_layer_bench`, `exl3_ggml_cuda_real_router_ffn_bench`,
  `exl3_ggml_cuda_real_exl3_layer_bench`). The cache-backed bridge, which
  production uses, is unaffected.
- Measured on the external-routed bench (topk=10, shared 3090) against the old
  kernels: 323 -> 132 us at one row and 1115 -> 436 us at four rows. Bridged
  all-48-layer decode on the 3090: 3.77 -> ~4.1 tok/s (64 tokens, ABBA).
- Timings in `exl3-wait-policy-2026-10-05.md` predate this and used the old
  kernels.
