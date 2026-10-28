# ZeroMoE

**A zero-dependency C++ inference engine using NVMe-to-VRAM streaming and CUDA Graphs to run massive MoE models on consumer GPUs.**

"Zero-dependency" means no Python runtime, no PyTorch, and no external model
server to install: the build vendors ggml, nlohmann/json, and cpp-httplib at
pinned revisions and produces one native binary.

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
git clone https://github.com/ray-x/ZeroMoE.git
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
- **llama.cpp is the correctness oracle.** [`scripts/validate_vs_llama.sh`](scripts/validate_vs_llama.sh) runs an unpatched llama.cpp build on the same GGUF and compares per-step logits: `cosine`, `max_abs`, `rmse`, and the top-1 token. See [Logits against llama.cpp](docs/validation.md#2-logits-against-llamacpp-the-oracle-for-a-new-model).
- **llama.cpp is the performance reference.** A model is not accepted until steady-state decode reaches at least 90% of llama.cpp on the same GGUF, GPU, context, prompt length, and thread count; see `AGENTS.md` and `tools/check_full_graph_moe_perf.py`.

## Full CUDA graph and host-node offload

The CUDA path is graph-only. Each supported family registers a `FullGraphModel` adapter that builds a whole-model ggml graph, and there is no layer-by-layer expert executor behind it. If the graph does not fit, the adapter must report `Insufficient` with required/available byte counts before layered fallback is allowed; unsupported architectures fail startup.

On top of ggml's CUDA graph capture, ZeroMoE adds:

- **Conditional device subgraphs.** Resident and staged expert branches are separate CUDA child graphs selected by device-side predicates in a parent graph.
- **Host nodes (`HostNode`).** A patched ggml CUDA backend expands a marker op into device-to-host copy → `cudaLaunchHostFunc` → host-to-device copy. This runs CPU work (cold-expert FFN, activation, or usage tracking) in the middle of a captured graph without leaving it. See [`patches/ggml-host-node/`](patches/ggml-host-node/) and [`models/qwen/ggml_qwen4exp_graph_host_node.cpp`](models/qwen/ggml_qwen4exp_graph_host_node.cpp).
- **Early-ready CPU lane.** Cold experts execute on the CPU in parallel with the GPU branch instead of serializing after it (`patches/ggml-host-node/0003-*`).
- **Staging and telemetry callbacks.** Host callbacks mark GPU stage boundaries and record routing/placement usage for the next request.

Host nodes are not permitted inside conditional bodies, and capture retains tensor addresses; the boundary and lifetime rules are in [docs/cuda-graph-api-review-2026-10-05.md](docs/cuda-graph-api-review-2026-10-05.md), with measured host-node behavior in [docs/runtime-settings.md](docs/runtime-settings.md). The adapter contract for adding a family is in [docs/design/full-graph-model-onboarding.md](docs/design/full-graph-model-onboarding.md).

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

EXL3 checkpoints (safetensors, `model_format = "exl3"`) run through the CPU or CUDA
matrix dispatch without a GGUF conversion: the decoder reads `mul1` weights directly
from safetensors, supports integer bitrates 1-8, and streams the selected routed
experts and PLE embedding rows from disk. Dense weights load lazily into host memory
and symlinked shards are accepted.

```bash
./build/zeromoe --config /path/to/qwen4exp_exl3_gpu.toml --prompt "Hello" --predict 8
```

The checkpoint directory has no HF `config.json`, so the profile uses
`metadata_model_path` to read architecture and tokenizer metadata from a matching
Qwen GGUF; its weight payloads are not used. Setup, kernel coverage, and same-split
benchmarking are in [docs/qwen-exl3-decode.md](docs/qwen-exl3-decode.md) and
[docs/exllamav3-backend.md](docs/exllamav3-backend.md).

### Running the engine

Launch the engine directly by pointing it at your local GGUF model, either via a TOML config (preferred) or legacy CLI flags:

```bash
./build/zeromoe --config configs/single_gpu.toml
# or, legacy CLI flags:
./build/zeromoe --model /path/to/model.gguf --ram 48.0 --ctx-size 4096
```

These are the flags most runs need. The full list - API serving, speculative
decoding, I/O and logging - is in [docs/cli-reference.md](docs/cli-reference.md).

| Flag | Description |
| --- | --- |
| `--config <file>` | Load a recipe from TOML (preferred over legacy flags). |
| `--model <path>` | GGUF file or shard directory (legacy CLI mode). |
| `--ram <GB>` | Host expert-memory budget. |
| `-c, --ctx-size <N>` | Context window (default 4096). |
| `-n, --predict <N>` | Maximum new tokens (default 1024). |
| `-t, --threads <N>` | CPU thread limit. |
| `--temp <F>` | Sampling temperature; `0` selects greedy. |
| `--main-gpu <ID>` | CUDA device for single-GPU mode. |
| `--serve` | Serve the OpenAI/Anthropic APIs. |
| `--help`, `-h` | Show the CLI help. |

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

## Advanced 2-bit quantization (IQ2_M)

ZeroMoE includes dedicated lookup vectors built explicitly to unlock and execute heavily compressed 2-bit IQ2_M model variations.

Rather than running heavy tensor conversions across your PCIe boundaries, the engine reads packed index fields directly from your storage blocks, unpacks scale registers inside memory-aligned cache steps, and feeds your execution pipelines with minimal latency.

## Validating a new model, or a change to a model's FFN flow

Throughput numbers say nothing about correctness: a broken graph can look faster.
Check correctness first, then speed. The full sequence - graph-selection gate,
unit checks, logits against llama.cpp, bisecting a mismatch, the before/after
regression witness, and where to record results - is in
[docs/validation.md](docs/validation.md).

## License

ZeroMoE is licensed under the GNU General Public License, version 3 only.
See [LICENSE](LICENSE) for the complete terms.

The engine builds on MIT-licensed third-party code (llama.cpp/ggml, ExLlamaV3,
nlohmann/json, and cpp-httplib) and contains ports of ggml quantization code.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the notices and
upstream licenses.
