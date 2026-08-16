# MoeStream

MoeStream is an ultra-fast, zero-dependency inference engine written from the ground up in native C++ and CUDA. It is specifically designed to stream frontier-class Mixture-of-Experts (MoE) GGUF models on consumer-grade hardware.

By implementing strict graph-level memory isolation, MoeStream maps and holds core persistent network blocks directly in physical VRAM and RAM, while dynamically streaming massive routed expert matrices directly from NVMe storage on an on-demand basis.

---

## Architectural Performance Foundations

- **Zero-Dependency GGUF Engine:** Automatically parses binary tensor offsets and extracts embedded BPE tokenizers straight from the GGUF file metadata header array. No external json or weight config maps required.
- **Lock-Free Asynchronous Prefetching:** Utilizes a highly optimized Single-Producer/Single-Consumer (SPSC) atomic ring buffer to dispatch prefetch instructions ahead of execution. It checks Layer $L$ activations against Layer $L+1$ router parameters to preload upcoming expert matrices via non-blocking DMA.
- **Fused AVX-512 / VNNI Compute Core:** Eliminates intermediate vector allocation overhead by dequantizing GGUF blocks directly inside physical CPU registers using hardware-accelerated integer dot products (`_mm512_dpbusd_epi32`).
- **RTX 3090 Integer Acceleration:** Features an advanced Ampere-native CUDA kernel leveraging the hardware-level `__dp4a` instruction alongside register warp-shuffles to handle dense projections instantly.
- **MLA Weight Absorption:** Reduces the operational scanning footprint of Multi-head Latent Attention layers from $O(\text{Context} \times H \times D)$ down to a compact $O(\text{Context} \times \text{kv\_lora})$ by calculating attention directly in the lower-dimensional latent space.

---

## 📂 Repository Layout Tree

```text
MoeStream/
├── CMakeLists.txt                 # Unified C++/CUDA compiler configuration
├── setup.sh                       # Automated installation and toolchain builder
├── README.md                      # Engine configuration documentation
├── include/                       # Core Architecture Headers
│   ├── engine.hpp                 # Memory allocation & expert slab managers
│   ├── gguf_loader.hpp            # Binary parsing & architecture discovery
│   ├── kernels_cpu.hpp            # AVX-512 block structures & primitives
│   ├── iq2_m_decoder.hpp          # 2-bit IQ2_M codebook dequantizers
│   ├── gguf_tokenizer.hpp         # Embedded vocabulary metadata extractor
│   ├── cuda_manager.hpp           # RTX Host-to-Device memory bridge
│   ├── async_prefetcher.hpp       # Lock-free SPSC prefetch queue
│   └── ram_manager.hpp            # MemAvailable monitor & LRU manager
├── src/                           # Core Implementation Units
│   ├── main.cpp                   # Entry point and interactive terminal CLI
│   ├── execution_layer.cpp        # Layer forward lookahead orchestration
│   ├── attention_layer.cpp        # Absorbed attention matrix calculation
│   ├── graph_initializer.cpp      # Topological parameter mapping core
│   ├── kernels_cpu.cpp            # x86 AVX-512 CPU kernel backend
│   └── kernels_cpu_neon.cpp       # Apple Silicon ARM NEON CPU kernel backend
├── cuda/                          # GPU Optimization Stack
│   └── gguf_cuda.cu               # Ampere __dp4a dynamic quantization kernels
└── api/                           # Production Network Gateways
    └── api_server.py              # Anthropic-compatible ASGI Python server

```

## Build and Installation Pipeline

### Linux + CUDA (full engine)

Run the automated environment setup tool. This script validates your system
compilers (`g++`, `nvcc`), prepares an isolated Python virtual environment,
installs ASGI gateway dependencies, and runs CMake to build the full engine:

```bash
chmod +x setup.sh
./setup.sh
```

Or manually:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### macOS / CPU-only build

The main `moestream` engine requires NVIDIA CUDA and cannot build on macOS.
However, the root `CMakeLists.txt` automatically detects Apple platforms and
switches to a CPU-only build that compiles:

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
./build-macos/gguf_tool --help
```

On Apple Silicon (M1/M2/M3/M4/M5) the CPU kernel backend uses ARM NEON +
SDOT for the f32 and integer-dot-product paths.  For optimized quantized
(MOE) expert decoding on macOS, build with ggml integration:

```bash
cmake -S . -B build -DMOESTREAM_USE_GGML=ON -DMOESTREAM_GGML_TAG=b3923
cmake --build build -j$(sysctl -n hw.ncpu)
```

### Interactive Terminal Generation

Launch the standalone engine directly by pointing it at your local GGUF model, either via a TOML config (preferred) or legacy CLI flags:

```bash
./build/moestream --config configs/single_gpu.toml
# or, legacy CLI flags:
./build/moestream --model /nvme/glm-5.2-iq2_m.gguf --ram 48.0 --ctx 4096
```

#### Command-Line Arguments

| Flag | Description |
| --- | --- |
| `--config <file>` | Load engine config from a TOML file (preferred over legacy CLI flags). |
| `--gen-config [output.toml]` | Generate a sample config file (default `moestream_config.toml`) and exit. |
| `--model <path>` | Path to a GGUF model file or directory (legacy CLI mode, used without `--config`). |
| `--ram <GB>` | Strict RAM budget in GB (default: auto-detect 88% of free system memory). |
| `--ctx <N>` | Maximum context size / token window (default: 4096). |
| `--ngen <N>` | Maximum new tokens to generate per response (default: 1024). |
| `--temp <F>` | Sampling temperature; `0.0` for greedy decode (default: 0.7). |
| `--top_p <F>` | Nucleus sampling top-p threshold (default: 0.90). |
| `--gpu <ID>` | CUDA device ID for single-GPU mode (default: 0). |
| `--drop-pages` | Drop expert pages from the OS page cache after use. |
| `--direct-io` | `O_DIRECT` expert-slab reads (Linux only; requires a model repacked with `tools/gguf_tool pack`). Falls back to buffered reads automatically if unsupported/unaligned. (`unpacked_expert_format` and `cuda_cufile` are TOML-config-only — see `--config`.) |
| `--io-uring` | Batch the async prefetcher's per-expert reads through `io_uring` (Linux only, kernel ≥ 5.6; independent of `--direct-io`). Falls back to sequential `pread`s automatically if unsupported. |
| `--prefetch-threads <N>` | Concurrent async prefetcher worker threads (default: 2). |
| `--verbose-moe` | Log every MoE host-cache hit (high volume, debug only). |
| `--promote-threshold <N>` | Number of host re-hits before an async host→device (H2D) promote (default: 1). |
| `--pin-threshold <N>` | VRAM-resident hit-count before permanently pinning an expert (never evicted; default: 8, `0` = off). |
| `--pin-file <file>` | Usage-frequency file for VRAM hot-set preload at startup (default: `.moestream_usage`). |
| `--expert-fetch-threads <N>` | OMP threads for parallel disk reads of a layer's top-k cache misses (default: 4). |
| `--raw-prompt` | Skip the chat-template wrap; tokenize stdin verbatim. Needed by `api_server.py` (which already applies its own formatting) and useful for token-count-fair benchmarking against other engines. |
| `--system-prompt <text>` | Prepend a system-role message ahead of the user turn (chat-template modes only; ignored with `--raw-prompt`). Empty by default — MoEStream never invents one on its own. Use this to steer behavior/language, e.g. `--system-prompt "Respond in English unless asked otherwise."` |
| `--log-path <file>` | Redirect logging to a file instead of stderr. |
| `--log-level <lvl>` | `VERBOSE` \| `DEBUG` \| `INFO` (default) \| `WARN` \| `ERROR`. |
| `--trace-filter <tags>` | Comma-separated tag prefixes (e.g. `HC,MLA`) restricting VERBOSE tensor tracing to just those checkpoints; requires `--log-level VERBOSE`. Empty (default) traces everything. See `include/tensor_trace.hpp` / CLAUDE.md §5. |
| `--trace-abort-on-nan` | Abort the process at the first NaN/Inf caught by a tensor-trace checkpoint, instead of continuing. Off by default. |
| `--help`, `-h` | Show the CLI help interface. |

> Most of these flags have a matching TOML key of the same name (e.g. `raw_prompt`, `system_prompt`, `pin_threshold`) for use with `--config`; a config file's values are used unless the equivalent CLI flag is also passed, in which case the CLI flag wins. Run `--gen-config` to see the full set of TOML-only options (e.g. `unpacked_expert_format`, `cuda_cufile`, `pin_strategy`) that have no CLI-flag equivalent.

### Deploying the Anthropic-Compatible API

MoeStream includes a production-grade FastAPI wrapper that maps external HTTP network streams straight into your local executable's stdin/stdout pipe channels. It matches the native Anthropic `/v1/messages` format scheme, serving Server-Sent Events (SSE) token chunks asynchronously.

To spin up the network gateway, activate your virtual workspace environment and run the application server:

```Bash
source api/venv/bin/activate
python3 api/api_server.py
```

Query the live inference endpoint utilizing standard cURL connection patterns:

```Bash
curl -X POST http://localhost:8000/v1/messages \
 -H "Content-Type: application/json" \
 -d '{
"model": "glm-5.2",
"max_tokens": 1024,
"temperature": 0.7,
"messages": [
{"role": "user", "content": "Explain the difference between thread concurrency and parallelism."}
]
}'
```

## Testing

MoeStream ships two test harnesses, split by what hardware they need.

### Full engine suite (Linux + CUDA + AVX-512)

`src/main_test.cpp` exercises the real CUDA DP4A / fused kernels, AVX-512
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
MoeStream includes dedicated lookup vectors built explicitly to unlock and execute heavily compressed 2-bit IQ2_M model variations.

Rather than running heavy tensor conversions across your PCIe boundaries, the engine reads packed index fields directly from your storage blocks, unpacks scale registers inside memory-aligned cache steps, and feeds your execution pipelines with minimal latency.
