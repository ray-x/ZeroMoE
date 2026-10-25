# ZeroMoE

ZeroMoE is an ultra-fast, zero-dependency inference engine written from the ground up in native C++ and CUDA. It is specifically designed to stream frontier-class Mixture-of-Experts (MoE) GGUF models on consumer-grade hardware.

By implementing strict graph-level memory isolation, ZeroMoE maps and holds core persistent network blocks directly in physical VRAM and RAM, while dynamically streaming massive routed expert matrices directly from NVMe storage on an on-demand basis.

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
└── api/                           # Optional API gateway (not included in this snapshot)

```

## Build and Installation Pipeline

### Linux + CUDA (full engine)

Run the automated setup tool. It validates the required compilers (`g++`,
`nvcc`) and builds the engine. Optional API dependencies are installed only
when the API server and its requirements file are present:

```bash
chmod +x setup.sh
./setup.sh
```

Or manually:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DZEROMOE_USE_GGML=ON
cmake --build build -j$(nproc)
```

AVX-512 VNNI is enabled by default for CPU performance. On a CPU without
AVX-512 VNNI, compiling with it enabled may cause an illegal-instruction
failure. Configure with `-DZEROMOE_ENABLE_AVX512_VNNI=OFF` to disable VNNI in
both ZeroMoE and ggml CPU kernels.

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
cmake -S . -B build -DZEROMOE_USE_GGML=ON -DZEROMOE_GGML_TAG=b3923
cmake --build build -j$(sysctl -n hw.ncpu)
```

#### ggml-cuda / full-graph decode build

The experimental ggml-backed full-graph prefill/decode path (`ZEROMOE_QWEN4EXP_FULL_GRAPH*` env vars at runtime) requires linking ggml's CUDA backend:

```bash
cmake -S . -B build -DZEROMOE_USE_GGML=ON -DZEROMOE_USE_GGML_CUDA=ON
cmake --build build -j$(nproc)
```

`ZEROMOE_ENABLE_CUDA_GRAPHS` (default **ON**) enables ggml's CUDA graph capture/replay for this path. ggml's own default for `GGML_CUDA_GRAPHS` is OFF for any consumer other than llama.cpp itself, so without this, a plain `ZEROMOE_USE_GGML_CUDA=ON` build silently compiles out CUDA graph capture — every decode step then falls back to launching each graph node's kernel individually instead of replaying one captured graph per split. Measured impact on this project's Qwen4Exp full-graph decode: **~20 tok/s with graphs enabled vs. ~8 tok/s silently disabled**, on identical hardware/model/config. Opt out with `-DZEROMOE_ENABLE_CUDA_GRAPHS=OFF` if you need to A/B against the non-graph path.

#### Runtime settings (host-node mode, CPU threads, CUDA schedule)

`ZEROMOE_QWEN4EXP_HOST_NODE`, `ZEROMOE_QWEN_CPU_THREADS` and `ZEROMOE_CUDA_SCHEDULE` change how the Qwen4Exp CPU/GPU
decode path executes and interact with each other (host-node mode automatically selects a blocking CUDA wait, +24%). Values,
defaults, measurements and recommended combinations are in [docs/runtime-settings.md](docs/runtime-settings.md).


### Qwen EXL3 safetensors decode

EXL3 is experimental and its example TOML profiles are maintainer-local files
under the ignored `configs/testing/` directory; they are not included in the
public repository. Supply your own EXL3 profile to run these examples.

To reuse ExLlamaV3's **complete optimized execution backend**, including its
GPU attention/cache/decode and AVX-512/VNNI CPU expert worker, use
`configs/testing/qwen4exp_exllamav3_26_22.toml`. Setup, metadata requirements, and
same-split benchmarking are documented in [docs/exllamav3-backend.md](docs/exllamav3-backend.md).
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
./build_ggmlcuda/zeromoe --config configs/testing/qwen4exp_exl3_decode.toml \
    --prompt "Hello" --ngen 8
```

`configs/testing/qwen4exp_exl3_load.toml` retains the catalog/payload smoke mode.
For GPU projections, use a CUDA build and `configs/testing/qwen4exp_exl3_gpu.toml`:

```bash
./build_ggmlcuda/zeromoe --config configs/testing/qwen4exp_exl3_gpu.toml \
    --prompt "Hello" --ngen 8
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
synthetic matrix parity bound is 3% RMS. It is enabled in the supplied 26/22
profile; set it false for exact CPU matmuls. Unquantized F32
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
The default is false. `configs/testing/qwen4exp_exl3_gpu_hc_experiment.toml` enables
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
ZEROMOE_TEST_EXL3_ONLY=1 OMP_NUM_THREADS=4 ./build_ggmlcuda/zeromoe_tests
```

To benchmark the EXL3 CUDA kernels without opening a model, use synthetic
2560×640 expert matrices. The first command represents the 15 GPU expert
layers on device 0; the second represents seven layers on device 1:

```bash
make bench-exl3 BUILD_DIR=build_ggmlcuda ARGS="--device 0 --layers 15 --experts 10 --bits 3"
make bench-exl3 BUILD_DIR=build_ggmlcuda ARGS="--device 1 --layers 7 --experts 10 --bits 3"
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
make bench-exl3-cpu BUILD_DIR=build_ggmlcuda ARGS="--threads 6 --bits 3"
make bench-exl3-cpu BUILD_DIR=build_ggmlcuda ARGS="--threads 12 --bits 3"
```

HyperConnection dispatch has a model-free benchmark with the checkpoint's
width 2560, four streams and rank 320:

```bash
make bench-exl3-hc BUILD_DIR=build_ggmlcuda ARGS="--device 0 --blocks 97 --iterations 10"
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
include routed FFN calls and reused eviction buffers. The supplied profile uses
the matching 26-CPU/22-GPU routed-expert split, with GPU layers 26–40 on the
3090 and 41–47 on the 5060 Ti. The profiles use 12 OMP threads. A synthetic
26-layer CPU expert benchmark measured 42.9 ms at 12 threads versus 55.0 ms
at six threads before the packed-tile grouping optimization. The int8 VNNI
path now shares activation/decode setup across four adjacent tiles for
bitrates 1–4 and two tiles for bitrates 5–8. Verify these changes with the full decoder before treating
that isolated gain as end-to-end throughput. The remaining attention/GDN and orchestration
path still runs through the CPU decoder rather than Bee's full GPU graph.

### GGUF generation

Launch the standalone engine directly by pointing it at your local GGUF model, either via a TOML config (preferred) or legacy CLI flags:

```bash
./build/zeromoe --config configs/single_gpu.toml
# or, legacy CLI flags:
./build/zeromoe --model /nvme/glm-5.2-iq2_m.gguf --ram 48.0 --ctx 4096
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
| `--pin-file <file>` | Usage-frequency file for VRAM hot-set preload at startup (default: `.zeromoe_usage`). |
| `--expert-fetch-threads <N>` | OMP threads for parallel disk reads of a layer's top-k cache misses (default: 4). |
| `--raw-prompt` | Skip the chat-template wrap; tokenize stdin verbatim. Needed by `api_server.py` (which already applies its own formatting) and useful for token-count-fair benchmarking against other engines. |
| `--system-prompt <text>` | Prepend a system-role message ahead of the user turn (chat-template modes only; ignored with `--raw-prompt`). Empty by default — ZeroMoE never invents one on its own. Use this to steer behavior/language, e.g. `--system-prompt "Respond in English unless asked otherwise."` |
| `--log-path <file>` | Redirect logging to a file instead of stderr. |
| `--log-level <lvl>` | `VERBOSE` \| `DEBUG` \| `INFO` (default) \| `WARN` \| `ERROR`. |
| `--trace-filter <tags>` | Comma-separated tag prefixes (e.g. `HC,MLA`) restricting VERBOSE tensor tracing to just those checkpoints; requires `--log-level VERBOSE`. Empty (default) traces everything. See `core/tensor_trace.hpp` / CLAUDE.md §5. |
| `--trace-abort-on-nan` | Abort the process at the first NaN/Inf caught by a tensor-trace checkpoint, instead of continuing. Off by default. |
| `--help`, `-h` | Show the CLI help interface. |

> Most of these flags have a matching TOML key of the same name (e.g. `raw_prompt`, `system_prompt`, `pin_threshold`) for use with `--config`; a config file's values are used unless the equivalent CLI flag is also passed, in which case the CLI flag wins. Run `--gen-config` to see the full set of TOML-only options (e.g. `unpacked_expert_format`, `cuda_cufile`, `pin_strategy`) that have no CLI-flag equivalent.

### Optional API gateway

The API server is not included in this repository snapshot. `setup.sh` skips
API dependency installation unless both `api/api_server.py` and
`api/requirements.txt` are present.

## Testing

ZeroMoE ships two test harnesses, split by what hardware they need.

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
ZeroMoE includes dedicated lookup vectors built explicitly to unlock and execute heavily compressed 2-bit IQ2_M model variations.

Rather than running heavy tensor conversions across your PCIe boundaries, the engine reads packed index fields directly from your storage blocks, unpacks scale registers inside memory-aligned cache steps, and feeds your execution pipelines with minimal latency.

## Validating a new model, or a change to a model's FFN flow

Throughput numbers say nothing about correctness: a broken graph can look faster. Check correctness first, then
speed. Use the layers below from cheapest to most expensive; a change to attention, routing, the MoE/FFN block,
quant handling or expert placement should pass all that apply.

### 1. Unit and structure checks (seconds, no model)

```bash
make quant_table_test graph_settings_test layer_plan_test    # in your build dir
./quant_table_test && ./graph_settings_test tests/graph_settings_defaults.snapshot && ./layer_plan_test
make moe_conformance                                         # shared-code rules (no family checks, no direct env reads)
```

### 2. Logits against llama.cpp (the oracle for a new model)

llama.cpp runs the same GGUF on the CPU; ZeroMoE dumps the raw decode logits (before repetition penalty and
sampling) with `ZEROMOE_DECODE_LOGITS_DUMP`. `tools/ling_llama_logits.cpp` is the reference harness (greedy,
one logits row per step, same binary row format) and `tools/compare_logits.py` compares the two files.

```bash
scripts/validate_vs_llama.sh <model.gguf> <zeromoe.toml> "The capital of France is" 8 ./build_master/zeromoe
# env: LLAMA_SRC, LLAMA_BUILD (an UNPATCHED llama.cpp build), DEVS, MS_ENV="ZEROMOE_X=..." for model-specific knobs
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
- `ZEROMOE_MIMO2_NODE_TRACE=2` (MiMo) reports the first non-finite node; `trace_abort_on_nan = true` in the TOML
  (see `include/tensor_trace.hpp`) does it for the layered path. Do not time a run with a trace mode on: it
  disables the early-ready bridge.
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
