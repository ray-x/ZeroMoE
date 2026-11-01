# ZeroMoE

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![CI](https://github.com/ray-x/ZeroMoE/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/ray-x/ZeroMoE/actions/workflows/ci.yml)

ZeroMoE is a high-performance C++/CUDA inference engine optimized for full-graph MoE execution and AVX-512 VNNI, designed to run large open-weight MoE models locally. It uses tiered storage and compute resources to fit models within local RAM and VRAM limits, supporting multiple popular open-weight MoE families.

Linux with NVIDIA CUDA is the main target. The CPU engine also runs on macOS. Windows support is still in progress.

## Build

On Linux with an NVIDIA GPU and CUDA toolkit:

```bash
./setup.sh
```

Or build with CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The CUDA engine requires an x86-64 CPU with AVX-512 F/CD/BW/DQ/VL. AVX-512 VNNI is enabled when the build host supports it; `-DGGML_AVX512_VNNI=OFF` disables it. The first build fetches pinned dependencies, so it needs network access. Clone the repository with Git rather than downloading a source ZIP; the build checks its Git provenance.

On macOS, build the CPU engine and run the component tests with:

```bash
make build
make test-components
```

The macOS executable is `build-macos/zeromoe_cpu`. See [the configuration guide](docs/configuration.md) for more build and configuration options.

Windows support is in progress. The documented build paths currently cover Linux CUDA and macOS CPU; there are no native Windows build instructions yet.

## Run a model

ZeroMoE does not include or download model weights. Put a supported GGUF outside the repository, then run:

```bash
./build/zeromoe --model /path/to/model.gguf --prompt "Hello"
```

For repeatable runs, copy the starter profile and set `model_path`:

```bash
cp configs/starter.toml my-model.toml
./build/zeromoe --config my-model.toml --prompt "Hello"
```

Check [supported models and their profiles](docs/models/README.md) before choosing a model. The profiles under [`configs/`](configs/) may also need local paths for mirrors, expert stores, or usage data. See the [configuration guide](docs/configuration.md) for TOML setup. Run `./build/zeromoe --help` or use the [CLI reference](docs/cli-reference.md) for command-line options.

## How CUDA MoE works

CUDA MoE models run through the shared full-graph runtime. Each supported GGUF architecture and tensor layout needs a registered model adapter; a family name alone does not guarantee support. The engine checks the adapter and plans graph memory before running. An unsupported model fails at startup, and a graph build or execution failure is fatal for that request. Memory fallback is allowed only when the planner finds that the graph does not fit and the model supports fallback.

- [Model support and setup](docs/models/README.md)
- [Full-graph design and memory rules](docs/design/full-graph-model-onboarding.md)
- [Runtime settings](docs/runtime-settings.md)
- [Benchmark profiles and measurements](configs/bench/README.md)
- [EXL3 safetensors setup](docs/qwen-exl3-decode.md)
- [EXL3 safetensors backend guide](docs/exllamav3-backend.md)

## Model family support and benchmarks

### Hardware setup
- **CPU:** Xeon W-2235 (6 cores / 12 threads, AVX-512 with VNNI)
- **RAM:** 125 GB DDR4
- **GPUs:** RTX 3090 (24 GB, PCIe Gen3 x16) + RTX 5060 Ti (16 GB, PCIe Gen3 x8)
- **Storage:** NVMe + SATA SSDs (for offload), zram swap only
- **Software:** Linux (CachyOS), CUDA 13.3, driver 610.57.04

### Supported model families and peak performance

| Model family | Quant / GGUF used in tests | Hardware | Prefill (peak) | Decode (peak seek/speed) | Notes |
|---|---|---|---|---|---|
| **GLM 5.3 Flash** | AJ IQ2_XXS / GSQ-RCO Q2_K/Q3_K/Q4_K (3.0-bit) | 3090 + 5060 Ti | **272.7 tok/s** (4090 tokens, AJ IQ2_XXS) | **15.84 tok/s** (in-sample, GSQ-RCO fitted ranking) | Mixed quant variants tested; prefill/decode tuned differently; needs retest/tuning for consistent peak numbers. See [perf-baselines](configs/bench/perf-baselines.md). |
| **GLM 5.3 DSA** | UD-IQ2_M | 3090 + 5060 Ti | **12.95 tok/s** (426 tokens) | **2.06 tok/s** (first-to-last, with retarget) | SSD-bandwidth bound full-graph path. |
| **Qwen 3.8 Flash** | GSQ-RCO Q2_0 / UD-IQ1_M / Q2_0 (hybrid) (+ MTP draft Q4_K_M for MTP runs) | 3090 only | **548 tok/s** (5895 tokens, UD-IQ1_M) | **83.0 tok/s** (with MTP, GSQ-RCO Q2_0) | Merged across Q2_0/IQ1_M variants; MTP gives up to 1.72x speedup. See [perf-baselines](configs/bench/perf-baselines.md). |
| **DeepSeek V4.1 Flash** | See [bench config](configs/bench/deepseek_v41_100gib_ddr_17gib_vram_dual_gpu.toml) | 3090 + 5060 Ti | ~29 tok/s (817 tokens) | **4.0-4.5 tok/s** | Dual-GPU offload. |
| **MiMo V2.6 Flash** | See [bench config](configs/bench/mimo26_flash_gsq_rco_3bit_ingraph_dual.toml) | 3090 + 5060 Ti | **80.4 tok/s** (1022 tokens) | **13.4-15.7 tok/s** | Dual-GPU, in-graph decode. |
| **Ling** | See [bench config](configs/bench/ling/) | 3090 + 5060 Ti | **22.2 tok/s** (1771 tokens) | **10.1 tok/s** | Full-graph prefill. |

*All numbers are from [configs/bench/perf-baselines.md](configs/bench/perf-baselines.md) on the hardware above. Performance varies by quantization, placement, and run conditions. See individual model docs in [docs/models/](docs/models/) for specific setup.*

## Development

Run `make test-components` for the CPU component suite. On Linux with CUDA, `make test` runs the full engine suite. See [the tools guide](tools/README.md) for GGUF packing and verification, [validation](docs/validation.md) for model correctness checks, and [performance measurement](docs/performance-measurement.md) for benchmark practice.

The repository uses [`.clang-format`](.clang-format) for C++, CUDA, and headers. Format files as you change them; CI checks changed first-party files.

The engine also has an OpenAI and Anthropic compatible API server. See [the API server guide](docs/api-server.md) for setup and current limits.

## License

GNU General Public License v3 only. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
