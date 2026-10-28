# ZeroMoE

ZeroMoE is a native C++23/CUDA inference engine for Mixture-of-Experts (MoE) GGUF models. It keeps dense weights and KV state resident while streaming routed expert matrices from NVMe on demand, so models far larger than VRAM can run on consumer hardware.

It is **not a fork of llama.cpp** and does not link `libllama`. It builds on **ggml**, the tensor library underneath llama.cpp, from a pinned upstream commit, and adds a whole-model execution graph, expert streaming, and graph-level host/device offload on top.

- **GGUF-native loader:** parses binary tensor offsets and the embedded BPE tokenizer straight from the GGUF metadata; no separate JSON or weight config maps.
- **Lock-free asynchronous prefetching:** an SPSC atomic ring buffer checks layer $L$ activations against layer $L+1$ router parameters to preload upcoming expert matrices via non-blocking DMA.
- **Fused AVX-512 / VNNI compute core:** dequantizes GGUF blocks directly in CPU registers using hardware integer dot products (`_mm512_dpbusd_epi32`).
- **RTX 3090 integer acceleration:** an Ampere-native CUDA kernel using `__dp4a` and register warp-shuffles for dense projections.
- **MLA weight absorption:** calculates Multi-head Latent Attention directly in the lower-dimensional latent space, reducing the attention scanning footprint.

---

## Requirements

| Component | Requirement |
|---|---|
| OS | Linux x86-64 for the CUDA engine. macOS (Apple Silicon) builds the CPU-only tools and tests. |
| GPU | NVIDIA GPU supported by the installed CUDA toolkit. Tested on RTX 3090 (`sm_86`) and RTX 5060 Ti (`sm_120a`). |
| CUDA | CUDA Toolkit and driver. The reference build uses CUDA 13.3.73 with driver 610.57.04. |
| CPU | x86-64 with AVX-512 F/CD/BW/DQ/VL. AVX-512 VNNI is on by default; disable it with `-DGGML_AVX512_VNNI=OFF` on CPUs without VNNI. |
| RAM | 64 GB or more recommended. Host RAM is used as the expert cache; test profiles assume 125 GiB. |
| Disk | NVMe SSD. Keep the GGUF and any repacked stores outside the repository. |
| Build tools | CMake ≥ 3.25, GNU Make, `git`, GNU `patch`, Python 3, and a C++23-capable host compiler (tested with GCC 16.2 and nvcc 13.3). |

**ggml is required.** Every CUDA build compiles and links ggml's CUDA backend. You do not install ggml yourself: CMake fetches it from pinned llama.cpp commit `8baabd53b77d1c48c6ef8d3b620436292f3f6231` (ggml 0.27.0), copies it into the build tree, and applies ZeroMoE's patch series. A normal build also fetches pinned `nlohmann/json` and `cpp-httplib` for the built-in API server. Model weights are never downloaded.

## Quick start

```bash
git clone <repository-url> ZeroMoE
cd ZeroMoE
./setup.sh
```

`setup.sh` checks the toolchain, then configures and builds a Release binary at
`build/zeromoe` (on macOS it builds the CPU-only tools in `build-macos/`). It
fetches the pinned ggml source on the first run, so that step needs network
access.

Point a starter profile at your GGUF, then run:

```bash
$EDITOR configs/qwen38-flash.toml   # set model_path; adjust budgets for your hardware
./build/zeromoe --config configs/qwen38-flash.toml --prompt "Hello" --predict 32
```

- Supported families and profiles: [docs/models/README.md](docs/models/README.md)
- Manual build commands and options: [Build and installation](#build-and-installation)
- Configuration keys and precedence: [docs/configuration.md](docs/configuration.md)
- Environment variables: [docs/env-vars.md](docs/env-vars.md)

> **Clone with git.** GitHub "Download ZIP" archives contain neither the pinned ggml sources nor a git revision, and the CUDA build stops at its build-provenance check. A normal `git clone` builds.

### Getting model weights

ZeroMoE does not bundle or download weights. Converted GGUFs for the supported
families are published on Hugging Face by their respective authors; download the
quantization you want with your preferred client and keep it outside the
repository:

```bash
# Replace with the repository and file you have access to.
hf download <org>/<repo> <model>.gguf --local-dir ~/models
```

Set `model_path` in the profile to that file, or to a directory containing split
shards (`*-00001-of-0000N.gguf`). At startup the engine checks the exact
`general.architecture` and tensor topology against its registered
`FullGraphModel` adapters and refuses an unsupported model.

For `--direct-io`, repack the model into the per-expert layout first, then point
`model_path` at the output directory:

```bash
./build/gguf_tool pack model.gguf ./model-packed --verify-all
```

See [tools/README.md](tools/README.md) for packing, verification, and the I/O
benchmark.

## Supported models

| Model | GGUF `general.architecture` | Starter profile |
|---|---|---|
| Qwen 3.8 Flash | `qwen4exp` | [`configs/qwen38-flash.toml`](configs/qwen38-flash.toml) |
| Qwen3.6 35B-A3B | `qwen35moe` | [`configs/bench/qwen36_35b_a3b_iq2m_5060_tg400.toml`](configs/bench/qwen36_35b_a3b_iq2m_5060_tg400.toml) |
| GLM 5.3 Flash | `glm5next` | [`configs/glm53-flash.toml`](configs/glm53-flash.toml) |
| Ling 3.1 Flash | `bailingmoe3` | [`configs/ling31-flash.toml`](configs/ling31-flash.toml) |
| DeepSeek 4.1 Flash | `deepseek41` | [`configs/deepseek41-flash.toml`](configs/deepseek41-flash.toml) |
| GLM 5.3 | `glm-dsa` | [`configs/glm53-dsa.toml`](configs/glm53-dsa.toml) |
| MiMo V2.6 Flash RL (experimental) | `mimo2` | [`configs/mimo26-flash.toml`](configs/mimo26-flash.toml) |

Exact quantizations, measured decode bands, and per-model gotchas are in [docs/models/README.md](docs/models/README.md). A model is only supported when its exact `general.architecture` and tensor topology have a registered `FullGraphModel` adapter; a shared family label is not proof of compatibility.

## Relationship to llama.cpp and ggml

ZeroMoE borrows heavily from llama.cpp's design while keeping its own runtime:

- **ggml is the backend layer.** The CUDA engine always builds and links ggml's CUDA backend. ggml is fetched from a pinned llama.cpp commit and patched at configure time; the submodule is copied, never modified. See [docs/build-ggml.md](docs/build-ggml.md) and [patches/README.md](patches/README.md).
- **Formats follow ggml.** GGUF parsing, quantization block layouts (Q2_0, IQ1_S, IQ2_*, MXFP4, ...), and CPU reference dequantization use ggml's definitions. The IQ2 dequantizer in `kernels/quant/iq2_matmul.cu` is a port of ggml's `dequantize_row_iq2_*`.
- **Full CUDA graph design.** The engine follows llama.cpp's approach of letting ggml's CUDA backend capture pure-CUDA graph spans and replay them, then extends it with a whole-model parent graph and graph-level offload (next section).
- **Partial CLI compatibility.** `--ctx-size`, `--predict`, `--threads`, `--main-gpu`, `--log-file`, and `--verbosity` match llama.cpp names. ZeroMoE is not a drop-in replacement for `llama-cli`.
- **llama.cpp is the correctness oracle.** [`scripts/validate_vs_llama.sh`](scripts/validate_vs_llama.sh) runs an unpatched llama.cpp build on the same GGUF and compares per-step logits: `cosine`, `max_abs`, `rmse`, and the top-1 token. See [Logits against llama.cpp](#2-logits-against-llamacpp-the-oracle-for-a-new-model).
- **llama.cpp is the performance reference.** A model is not accepted until steady-state decode reaches at least 90% of llama.cpp on the same GGUF, GPU, context, prompt length, and thread count; see `AGENTS.md` and `tools/check_full_graph_moe_perf.py`.

## Full CUDA graph and host-node offload

The CUDA path is graph-only. Each supported family registers a `FullGraphModel` adapter that builds a whole-model ggml graph, and there is no layer-by-layer expert executor behind it. If the graph does not fit, the adapter must report `Insufficient` with required/available byte counts before layered fallback is allowed; unsupported architectures fail startup.

On top of ggml's CUDA graph capture, ZeroMoE adds:

- **Conditional device subgraphs.** Resident and staged expert branches are separate CUDA child graphs selected by device-side predicates in a parent graph.
- **Host nodes (`HostNode`).** A patched ggml CUDA backend expands a marker op into device-to-host copy → `cudaLaunchHostFunc` → host-to-device copy. This runs CPU work (cold-expert FFN, activation, or usage tracking) in the middle of a captured graph without leaving it. See [`patches/ggml-host-node/`](patches/ggml-host-node/) and [`models/qwen/ggml_qwen4exp_graph_host_node.cpp`](models/qwen/ggml_qwen4exp_graph_host_node.cpp).
- **Early-ready CPU lane.** Cold experts execute on the CPU in parallel with the GPU branch instead of serializing after it (`patches/ggml-host-node/0003-*`).
- **Staging and telemetry callbacks.** Host callbacks mark GPU stage boundaries and record routing/placement usage for the next request.

Host nodes are not permitted inside conditional bodies, and capture retains tensor addresses; the boundary and lifetime rules are in [docs/cuda-graph-api-review-2026-10-05.md](docs/cuda-graph-api-review-2026-10-05.md), with measured host-node behavior in [docs/runtime-settings.md](docs/runtime-settings.md). The adapter contract for adding a family is in [docs/design/full-graph-model-onboarding.md](docs/design/full-graph-model-onboarding.md).

---

## 📂 Repository Layout Tree

```text
ZeroMoE/
├── CMakeLists.txt                 # Unified C++/CUDA compiler configuration
├── setup.sh                       # Automated installation and toolchain builder
├── README.md                      # Engine configuration documentation
├── core/                          # Engine configuration, scheduling and execution
├── memory/                        # I/O, RAM and VRAM cache tiers
├── models/                        # Shared graph and model-family builders
├── backends/                      # CPU, CUDA, ggml and EXL3 backends
├── kernels/                       # CUDA kernels
├── tools/                         # GGUF tools
├── tests/                         # Native and CPU component tests
├── bench/                         # Benchmark scripts and fixtures
├── configs/                       # One portable starter profile per supported model
│   └── bench/                     # Benchmark-only profiles and published measurements
├── scripts/                       # Build and validation helpers
├── patches/                       # ggml patch series
├── docs/                          # Design notes and model recipes
├── server/                        # Native C++ OpenAI/Anthropic API server
└── THIRD_PARTY_NOTICES.md         # Licenses for bundled/ported third-party code

```

## Build and installation

### Linux + CUDA (full engine)

Run the automated setup tool. It checks the toolchain (`cmake`, `git`, `patch`,
`python3`, `c++`, `nvcc`), configures the pinned ggml source and patch series,
and builds the engine together with the native C++ HTTP server:

```bash
chmod +x setup.sh
./setup.sh
```

Or manually:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

For the smaller configuration surface (Release CPU/CUDA presets, normal versus
advanced help, canonical `graph_*` keys and legacy compatibility), start with
the [configuration guide](docs/configuration.md).

AVX-512 VNNI is enabled by default for CPU performance. On a CPU without
AVX-512 VNNI, compiling with it enabled may cause an illegal-instruction
failure. Configure with `-DGGML_AVX512_VNNI=OFF` to disable VNNI in
both ZeroMoE and ggml CPU kernels.

### OpenAI and Anthropic APIs

The C++ server is always built and exposes Chat Completions and Messages,
including SSE, Qwen tool calls, cancellation, and bounded request queues. It
runs in the engine process with one inference owner; no Python gateway is
needed.

See [API server](docs/api-server.md) for CPU/CUDA build commands, authentication,
examples, feature limits, and the experimental opt-in CPU prompt-state cache.
Native CUDA prompt caching is not yet available for API requests.

Before inference, edit a profile's `model_path` and any related model,
mirror, expert-store, or usage-profile paths for resources available on your
machine. Model weights are not included; example profiles use placeholders.
Keep model files and generated stores outside the repository.

The top-level `configs/` directory contains one starter profile per supported
model. Benchmark-specific profiles and the consolidated performance report
are in [`configs/bench/`](configs/bench/README.md). Additional local tuning
profiles may remain in `configs/` on your machine; they are ignored by Git.

### macOS / CPU-only build

The main `zeromoe` engine requires NVIDIA CUDA and cannot build on macOS.
However, the root `CMakeLists.txt` automatically detects Apple platforms and
switches to a CPU-only build that compiles:

* `zeromoe_cpu` — CPU inference engine and API server
* `gguf_tool` — pack / verify / io-bench utility (cross-platform)
* `test_components` — CPU-only component and kernel tests

```bash
make build                       # configures into build-macos on Apple platforms
make test                        # run the CPU kernel / component tests
./build-macos/gguf_tool --help   # use the GGUF utility
```

Or, equivalently, with CMake directly:

```bash
cmake -S . -B build-macos -DCMAKE_BUILD_TYPE=Release
cmake --build build-macos -j$(sysctl -n hw.ncpu)
./build-macos/test_components
./build-macos/zeromoe_cpu --config /path/to/model.toml --prompt "Hello"
./build-macos/gguf_tool --help
```

On Apple Silicon (M1/M2/M3/M4/M5) the CPU kernel backend uses ARM NEON +
SDOT for the f32 and integer-dot-product paths. ggml integration for optimized
quantized (MoE) expert decoding is enabled by default on macOS.

`ZEROMOE_METAL` defaults ON on Apple Silicon: prefill-sized F32 matmuls
(`S >= 32`) are dispatched to Metal/MPS, while decode and small batches stay
on NEON.  Configure with `-DZEROMOE_METAL=OFF` for a NEON-only build.

#### ggml-cuda / full-graph decode build

Every CUDA build links ggml's CUDA backend, so the ggml-backed full-graph prefill/decode path is always available and is the default; no runtime switch is required:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
# or use the checked-in preset:
cmake --preset cuda && cmake --build --preset cuda
```

CUDA builds always enable ggml CUDA graph capture/replay. ggml's own default for `GGML_CUDA_GRAPHS` is OFF for consumers other than llama.cpp, which silently removes the captured-graph path and regresses this project's measured Qwen4Exp decode from about 20 tok/s to 8 tok/s. CMake forces the required setting ON; it is not a user-selectable build mode.

CUDA architecture selection defaults to automatic detection from
`nvidia-smi` before any CUDA targets, including ggml-cuda, are configured.
On a 3090 + 5060 Ti host, the detected list is `86;120a`. Set
`CUDA_ARCH='86;120a'` for the build scripts or pass
`-DCMAKE_CUDA_ARCHITECTURES='86;120a'` directly to CMake to override detection.
For a GPU-less build host, supply the architectures explicitly. Verify that a
5060 Ti benchmark binary contains `sm_120a`; an `sm_86`-only build is not a
valid performance comparison.

#### Runtime settings (host-node mode, CPU threads, CUDA schedule)

`ZEROMOE_QWEN4EXP_HOST_NODE`, `ZEROMOE_QWEN_CPU_THREADS` and `ZEROMOE_CUDA_SCHEDULE` change how the Qwen4Exp CPU/GPU
decode path executes and interact with each other (host-node mode automatically selects a blocking CUDA wait, +24%). Values,
defaults, measurements and recommended combinations are in [docs/runtime-settings.md](docs/runtime-settings.md).


### Qwen EXL3 safetensors decode

EXL3 is experimental. The example profiles are maintainer-local files under the
ignored `configs/testing/` directory and are not shipped; the paths below are
placeholders for your own profiles. Setup, metadata requirements, and same-split
benchmarking are documented in [docs/exllamav3-backend.md](docs/exllamav3-backend.md).

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
ExLlamaV3 backend. [The waiting-policy report](docs/exl3-wait-policy-2026-10-05.md)
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

### Running the engine

Launch the engine directly by pointing it at your local GGUF model, either via a TOML config (preferred) or legacy CLI flags:

```bash
./build/zeromoe --config configs/single_gpu.toml
# or, legacy CLI flags:
./build/zeromoe --model /path/to/model.gguf --ram 48.0 --ctx-size 4096
```

#### Command-Line Arguments

For Qwen full-graph inference, `qwen4exp_prefill_state_handoff = true` keeps
prefill GDN state on the GPU for decode and defers host readback until a layered
fallback needs it. The default is `false`. It retains the prefill scheduler's
activation workspace during decode (about 0.18 GiB in the latest 3090 pp23
test, plus reusable GPU zero-state buffers; size depends on prompt shape and
placement). Fresh prefills use immutable CUDA zero-state inputs instead of
clearing and uploading recurrent state on every request. KV and PLE host snapshots remain
available. Prompt caching, MTP, and compact expert staging use the host path.
`ZEROMOE_QWEN4EXP_PREFILL_STATE_HANDOFF=0` or `1` overrides the TOML setting.
See [the state-handoff A/B guide](tests/qwen4exp_state_handoff_howto.md) for
reproduction commands, measured timings, and the memory tradeoff.
Full-GPU Qwen graphs omit the artificial CPU staging node and compute the
prefill output projection on GPU. `tests/qwen4exp_bee_fullgpu_compare.sh`
checks warm prefill and matched first-to-last decode performance against Bee.

| Flag | Description |
| --- | --- |
| `--config <file>` | Load engine config from a TOML file (preferred over legacy CLI flags). |
| `--gen-config [output.toml]` | Generate a sample config file (default `zeromoe_config.toml`) and exit. |
| `--model <path>` | Path to a GGUF model file or directory (legacy CLI mode, used without `--config`). |
| `--ram <GB>` | Strict RAM budget in GB (default: auto-detect 88% of free system memory). |
| `-c, --ctx-size <N>` | Maximum context size / token window (default: 4096). |
| `-n, --predict <N>` | Maximum new tokens to generate per response (default: 1024). |
| `-t, --threads <N>` | CPU thread limit (default: automatic). |
| `--temp <F>` | Sampling temperature; `0.0` for greedy decode (default: 0.7). |
| `--top-p <F>` | Nucleus sampling top-p threshold (default: 0.90). |
| `--repeat-penalty <F>` | Repetition penalty (default: 1.0, disabled). |
| `--main-gpu <ID>` | CUDA device ID for single-GPU mode (default: 0). |
| `--serve` | Serve the OpenAI Chat Completions and Anthropic Messages APIs from the loaded model. |
| `--host <addr>` | API bind address (default `127.0.0.1`; a non-loopback bind requires `--api-key` or `ZEROMOE_API_KEY`). |
| `--port <port>` | API port (default 8080). |
| `--alias <name>` | API model name (default `zeromoe`). |
| `--api-key <key>` | API key checked on API routes; also read from `ZEROMOE_API_KEY`. |
| `--spec-type <none|draft-mtp>` | Speculative mode supported by ZeroMoE. |
| `--spec-draft-model <path>` | Optional external draft checkpoint. |
| `--spec-draft-n-max <N>` | Maximum speculative draft length (default: 3). |
| `--drop-pages` | Drop expert pages from the OS page cache after use. |
| `--direct-io` | `O_DIRECT` expert-slab reads (Linux only; requires a model repacked with `tools/gguf_tool pack`). Falls back to buffered reads automatically if unsupported/unaligned. (`unpacked_expert_format` is TOML-config-only — see `--config`.) |
| `--io-uring` | Batch the async prefetcher's per-expert reads through `io_uring` (Linux only, kernel ≥ 5.6; independent of `--direct-io`). Falls back to sequential `pread`s automatically if unsupported. |
| `--prefetch-threads <N>` | Concurrent async prefetcher worker threads (default: 2). |
| `--promote-threshold <N>` | Number of host re-hits before an async host→device (H2D) promote (default: 1). |
| `--pin-threshold <N>` | VRAM-resident hit-count before permanently pinning an expert (never evicted; default: 8, `0` = off). |
| `--pin-file <file>` | Usage-frequency file for VRAM hot-set preload at startup (default: `.zeromoe_usage`). |
| `--expert-fetch-threads <N>` | OMP threads for parallel disk reads of a layer's top-k cache misses (default: 4). |
| `--raw-prompt` | Skip the chat-template wrap; tokenize stdin verbatim. Useful for token-count-fair benchmarking and for clients that apply their own chat formatting. |
| `--system-prompt <text>` | Prepend a system-role message ahead of the user turn (chat-template modes only; ignored with `--raw-prompt`). Empty by default — ZeroMoE never invents one on its own. Use this to steer behavior/language, e.g. `--system-prompt "Respond in English unless asked otherwise."` |
| `--log-file <file>` | Redirect logging to a file instead of stderr (llama.cpp-compatible name). |
| `--verbosity <0..5>` | Set log threshold (llama.cpp-compatible numeric verbosity). |
| `--help`, `-h` | Show the CLI help interface. |

At DEBUG (`--verbosity 5`), numerical decode comparisons include
`[TEACHER-FORCE-LOGITS]` fingerprints and Qwen route-hit summaries use the
`[ROUTE-HIT]`, `[MOE-REQUIRED]`, and `[MOE-PROMPT]` headers. Filter these records from a verbose log by their header.

> Most of these flags have a matching TOML key of the same name (e.g. `raw_prompt`, `system_prompt`, `pin_threshold`) for use with `--config`; a config file's values are used unless the equivalent CLI flag is also passed, in which case the CLI flag wins. Run `--gen-config` to see the full set of TOML-only options (e.g. `unpacked_expert_format`, `pin_strategy`) that have no CLI-flag equivalent.

### Serving the model

The API server is compiled into `zeromoe`; there is no separate gateway and no
Python environment to install. Start it with `--serve`:

```bash
./build/zeromoe --config configs/qwen38-flash.toml --serve \
  --host 127.0.0.1 --port 8080 --alias zeromoe
```

See [docs/api-server.md](docs/api-server.md) for endpoints, authentication,
SSE, and current limits.

## Testing

ZeroMoE ships two test harnesses, split by what hardware they need.

### Full engine suite (Linux + CUDA + AVX-512)

`tests/native/main_test.cpp` exercises the real CUDA DP4A / fused kernels, AVX-512
dequantization against a reference, and cuBLAS BF16 dense matmul
(`[Test 1]`–`[Test 10]`). It cannot build on macOS (hard CUDA / AVX-512 deps).

```bash
make test
```

### CPU-only component tests (macOS and Linux)

`tests/test_components.cpp` is a single CPU-only translation unit covering the
pure-logic components — RoPE, routing/topk, GGUF size math, the SPSC ring, the
TOML parser, dequant oracles, BF16 round-trip, and O_DIRECT alignment-window
arithmetic. It builds with the system C++ compiler (no CMake, no CUDA), so it
runs on Apple Silicon too. This is the fast inner loop for changes that touch
CPU-testable code.

```bash
make test-components                       # all component tests
make test-components ARGS="rope"           # one test, by substring
make test-components ARGS="rope dequant"   # several (substring OR)
./tests/run_components.sh direct_io        # build+run the script directly
```

Filters are substring matches on the test name (`dequant` hits all four
dequant tests, `iq` hits the IQ-quant ones, etc.). Exit code `0` = pass,
`2` = no test matched the filter. Override the compiler with `CXX=g++` if
`c++` isn't the one you want.

### I/O benchmark (macOS and Linux)

`gguf_tool io` (`tools/gguf_io.cpp`) measures the streaming-layout improvement
from `gguf_tool pack` (`tools/gguf_pack.cpp`): it A/Bs the per-expert read
pattern — **3 scattered `pread`s** per expert on the original Unsloth packed
`*_exps` layout vs **1 tight `pread`** per expert on the repacked layout.
Cross-platform (no CMake / CUDA), with cache bypass handled inside the tool
(`F_NOCACHE` on macOS, `posix_fadvise(FADV_RANDOM)` on Linux). The original
model always reads plain-buffered; pass `--direct` to additionally re-read the
repacked shard through `O_DIRECT`/DMA (Linux only) so the real DMA benefit
shows up as a third line, not just the read-pattern delta.

```bash
# repack your model first (see tools/README.md), then:
make bench-io ARGS="/path/to/model.gguf ./repacked --reads 5000"
# or directly:
./tools/run_gguf_tool.sh io /path/to/model.gguf ./repacked --reads 5000 --direct
```

Use a **large real model** so reads hit disk — on a small fixture everything is
cached and the ratio is noise.

### Suggested workflow

Iterate on macOS with `make test-components ARGS="…"` for the CPU-testable
parts and `make bench-io ARGS="…"` to check the repack layout/read pattern,
then run the full `make test` on Linux before considering a change done.
Run `gguf_tool verify` on any repacked model directory before pointing the
engine at it — it's the pre-flight completeness gate (see `tools/README.md`).

---

⚙️ Advanced 2-Bit Quantization (IQ2_M)
ZeroMoE includes dedicated lookup vectors built explicitly to unlock and execute heavily compressed 2-bit IQ2_M model variations.

Rather than running heavy tensor conversions across your PCIe boundaries, the engine reads packed index fields directly from your storage blocks, unpacks scale registers inside memory-aligned cache steps, and feeds your execution pipelines with minimal latency.

## Validating a new model, or a change to a model's FFN flow

Throughput numbers say nothing about correctness: a broken graph can look faster. Check correctness first, then
speed. Use the layers below from cheapest to most expensive; a change to attention, routing, the MoE/FFN block,
quant handling or expert placement should pass all that apply.

### 0. CUDA MoE graph-selection gate

Before implementing CUDA kernels for a new MoE, trace its exact GGUF
architecture through `resolve_family_ops()` to the `FullGraphModel` registry.
The adapter's `plan_decode_memory()` must reject unsupported architectures or
tensor topologies before large allocations. Only an `Insufficient` result with
required/available byte counts and a reason permits layered decode; `Unknown`
requires graph decode, and `Unsupported` fails startup. Graph build/run failure
is not a memory plan.

For a full-fit hardware run, require graph decode to be enabled and exercised,
with zero legacy one-token forwards or per-expert dispatches:

```bash
python3 tools/check_full_graph_moe_run.py <run.log> --full-fit
```

For Qwen3.5/Qwen3.6 full-fit runs, also pass
`--require-all-expert-layers-on-device` to require the graph's placement proof
to report all expert layers resident on the device.

Coherent output is a smoke check, not a performance pass. For a same-GGUF,
same-GPU combined prompt/decode reference, save `llama-bench` JSON and enforce
the default 90% steady-decode bar with:

```bash
llama-bench -m /path/to/model.gguf -pg 16,400 -ngl 99 -dev CUDA1 \
    -mg 0 -sm none -t 10 -r 3 -o json > llama-bench.json
python3 tools/check_full_graph_moe_perf.py <run.log> llama-bench.json \
    --n-prompt 16 --n-gen 400 --device CUDA1
```

The ZeroMoE gate uses time spent in graph decode replays and reports graph
construction/upload separately. A result below the bar keeps onboarding in
progress; it is not accepted because generated text happens to look coherent.

Use `--allow-memory-fallback` only for a run whose log records an explicit
insufficient-memory plan and its reason.

### 1. Unit and structure checks (seconds, no model)

```bash
cmake --build build --target quant_table_test graph_settings_test layer_plan_test
./build/quant_table_test && ./build/graph_settings_test tests/graph_settings_defaults.snapshot && ./build/layer_plan_test
make moe_conformance                                         # shared-code rules (no family checks, no direct env reads)
```

### 2. Logits against llama.cpp (the oracle for a new model)

llama.cpp runs the same GGUF on the CPU; ZeroMoE dumps the raw decode logits (before repetition penalty and
sampling) with `ZEROMOE_DECODE_LOGITS_DUMP`. `tools/ling_llama_logits.cpp` is the reference harness (greedy,
one logits row per step, same binary row format) and `tools/compare_logits.py` compares the two files. Build the
reference from an **unpatched** llama.cpp checkout at the pinned commit `8baabd53b77d1c48c6ef8d3b620436292f3f6231`
(the harness uses that revision's `llama_batch_ext` API).

```bash
scripts/validate_vs_llama.sh <model.gguf> <zeromoe.toml> "The capital of France is" 8 ./build/zeromoe
# env: LLAMA_SRC, LLAMA_BUILD (an UNPATCHED llama.cpp build), DEVS, MS_ENV="ZEROMOE_<NAME>=..." for model-specific knobs
```

It builds the harness once (`build_logits_ref`), runs llama.cpp, then ZeroMoE with `--raw-prompt --temp 0`, and
prints one line per step:

```
step=3 vocab=152576 max_abs=0.41 rmse=0.07 cosine=0.99981 top1=374/374 top1_match=True
```

How to read it:
- **Step 0 is the prompt's last-token logits**: it isolates prefill + the whole stack, with no accumulated error.
  Steps 1+ follow each engine's own greedy choices, so once the two pick different tokens the rows are no longer
  comparable. Stop reading at the first `top1_match=False`.
- **Pass:** `top1_match=True` for every step before any divergence and `cosine` above about 0.999. `max_abs` and
  `rmse` are not zero: different kernels, different summation order, and (for quantized experts) different
  CPU/GPU quant paths. A cosine near 0.9 or a wrong first token is a real bug (wrong RoPE, router, scale, shared
  expert, or a quant type with the wrong block size).
- A near-tie can flip top-1 legitimately. Compare the logit gap of the two candidates in the dump before calling
  it a bug, and check a second prompt.
- Keep both prompts short at first (5-20 tokens) so a mismatch points at the stack, not at chunking.
- The machine must be idle: llama.cpp loads the whole model into RAM on the CPU.

### 3. Bisecting a mismatch

- Same graph, different placement: run once with everything on one GPU and once with the hot/cold split. If only
  one differs, the bug is in placement or the cold path, not the math (the MiMo NaN from a repeated placeholder id
  was found this way).
- `--graph-node-trace 2` (MiMo) reports the first non-finite node. Use `--verbosity 4` for tensor traces;
  capture to a file and filter the resulting log after the run.
- Check tensor sizes first: `./quant_table_test` and the `[PREFLIGHT]` lines at startup flag a tensor type that is
  missing from `memory/quant_table.hpp`.

### 4. Before/after a change to an existing model (the regression witness)

For a change inside one model's FFN or MoE flow, compare your tree against `master` on the **same prompt, greedy,
same hardware**, and require the generated tokens to match (`ZEROMOE_GENERATED_TOKEN_FILE=<file>` writes them):

```bash
bench/ling/reference-audit/gate.sh <binary> <tag>     # Ling: 32 tokens, [REQUEST-DECODE-OUTPUT] rate (baseline 10.07 tok/s)
bench/regression_gate.sh <binary> --models dsv41      # DeepSeek-V4.1, 64 tokens (band 3.5-4.6 tok/s)
bench/regression_gate.sh <binary> --models qwen38,glm53flash,ling31,dsv41,glm53   # all five, ~15 min
```

- Tokens must be identical when only structure changed (refactor). When the numerics changed on purpose (a new
  kernel, a different hot set), tokens may diverge after some steps; then read the text and run step 2.
- Speed: the first run after a fresh ggml build JIT-compiles for the sm_120 GPU and is 2-10x slow; use a warm run
  and an ABBA order (A, B, B, A), because the box's load moves results by several percent.
- Compare tokens only between binaries built from the same tree and ggml (`docs/build-ggml.md`).

### 5. Recording the result

Put the oracle comparison (prompt, steps, cosine range, first divergence) and the gate numbers in
`docs/models/<model>.md`, and update `configs/bench/perf-baselines.md` when a number improves or a band moves.

## License

ZeroMoE is licensed under the GNU General Public License, version 3 only.
See [LICENSE](LICENSE) for the complete terms.

The engine builds on MIT-licensed third-party code (llama.cpp/ggml, ExLlamaV3,
nlohmann/json, and cpp-httplib) and contains ports of ggml quantization code.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the notices and
upstream licenses.
