# ZeroMoE

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![CI](https://github.com/ray-x/ZeroMoE/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/ray-x/ZeroMoE/actions/workflows/ci.yml)

ZeroMoE is a C++/CUDA inference engine for running large mixture of experts (MoE) models locally. It uses tiered storage and compute resources to run models within local RAM and VRAM limits.

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

### Hardware setup (baseline)
- **CPU:** Xeon W-2235 (6 cores / 12 threads, AVX-512 with VNNI)
- **RAM:** 125 GB DDR4
- **GPUs:** RTX 3090 (24 GB, PCIe Gen3 x16) + RTX 5060 Ti (16 GB, PCIe Gen3 x8)
- **Storage:** NVMe + SATA SSDs (for offload), zram swap only
- **Software:** Linux (CachyOS), CUDA 13.3, driver 610.57.04

### Supported model families and peak performance

| Model family | GGUF quant / model file | Size / BPW | Hardware | Prefill (peak) | Decode (peak) | Notes |
|---|---|---|---|---|---|---|
| **GLM-5.3-Flash (AJ)** | `GLM-5.3-Flash-AJ-IQ2_XXS-0000{1,2}-of-00002.gguf` (IQ2_XXS/IQ2_S experts, Q6_K non-expert) | 87.35 GB / 2.230 bpw | 3090 + 5060 Ti | **272.7 tok/s** (4090 tokens) | **14.3 tok/s** (short prompt, 70.1 ms/tok) | Whole-layer offload; older code build. See [perf-baselines](configs/bench/perf-baselines.md). |
| **GLM-5.3-Flash (GSQ-RCO)** | `GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf` (Q2_K/Q3_K/Q4_K experts, Q8_0/BF16 non-expert) | 117.48 GB / 3.000 bpw | 3090 + 5060 Ti | 20-31 tok/s (short) / 5.7 tok/s (long, 2313 tokens) | **15.84 tok/s** (in-sample, fitted ranking) / **2.06 tok/s** (first-to-last, with retarget) | Offload pool 88 GiB; retarget eviction gives +16-18% decode. |
| **GLM-5.3 DSA (UD-IQ2_M)** | `GLM-5.3-DSA-UD-IQ2_M.gguf` (layered/graph paths) | Varies | 3090 + 5060 Ti | **12.95 tok/s** (426 tokens) | **1.91-2.06 tok/s** (with retarget, short in-distribution) / **1.52 tok/s** (long prompt) | SSD-bandwidth bound; full-graph path. |
| **Qwen 3.8 Flash Next** | `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2}-of-00002.gguf` (Q2_0 experts, IQ4_NL non-expert) + MTP draft `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` | 66.42 GB / 3.003 bpw | 3090 only | ~0.73-3.20 s TTFT | **83.0 tok/s** (decode, with MTP) / **50.0 tok/s** (plain) | MTP gives up to 1.72x speedup; promotion enabled by default with MTP. |
| **Qwen 3.8 Flash (UD-IQ1_M)** | `Qwen3.8-Flash-Next-UD-IQ1_M-0000{1,2,3}-of-00003.gguf` (IQ4_NL/IQ1_M/IQ2_XXS experts) | 74.54 GB / 3.370 bpw | 3090 only | **548 tok/s** (5895 tokens) | **28.66 tok/s** (split 22/26) / **43.12 tok/s** (hybrid hot experts) | Graph prefill with GPU op-offload. |
| **Qwen 3.8 Flash (Q2_0)** | `Qwen3.8-Flash-Next-Q2_0.gguf` (hybrid recipe) | ~66 GB / ~3.0 bpw | 3090 only | **438.9 tok/s** (1707 tokens) | **50.6 tok/s** (peak) / **46.0 tok/s** (steady) | Usage-ranked hot experts (14K/24K) in VRAM. |
| **DeepSeek V4.1 Flash** | `DeepSeek-V4.1-Flash-*.gguf` (see [bench config](configs/bench/deepseek_v41_100gib_ddr_17gib_vram_dual_gpu.toml)) | ~100 GB class | 3090 + 5060 Ti | ~29 tok/s (817 tokens) | **4.0-4.5 tok/s** | Dual-GPU offload. |
| **MiMo V2.6 Flash** | `MiMo-V2.6-Flash-*.gguf` (see [bench config](configs/bench/mimo26_flash_gsq_rco_3bit_ingraph_dual.toml)) | Varies | 3090 + 5060 Ti | **80.4 tok/s** (1022 tokens) | **13.4-15.7 tok/s** | Dual-GPU, in-graph decode. |
| **Ling** | `Ling-*.gguf` | Varies | 3090 + 5060 Ti | **22.2 tok/s** (1771 tokens) | **10.1 tok/s** | Full-graph prefill. |

*All numbers are from [configs/bench/perf-baselines.md](configs/bench/perf-baselines.md) on the hardware above. Performance varies by quantization, placement, and run conditions. See individual model docs in [docs/models/](docs/models/) for specific setup.*

## Development

Run `make test-components` for the CPU component suite. On Linux with CUDA, `make test` runs the full engine suite. See [the tools guide](tools/README.md) for GGUF packing and verification, [validation](docs/validation.md) for model correctness checks, and [performance measurement](docs/performance-measurement.md) for benchmark practice.

The repository uses [`.clang-format`](.clang-format) for C++, CUDA, and headers. Format files as you change them; CI checks changed first-party files.

The engine also has an OpenAI and Anthropic compatible API server. See [the API server guide](docs/api-server.md) for setup and current limits.

## License

GNU General Public License v3 only. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
