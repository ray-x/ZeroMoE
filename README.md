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
│   └── kernels_cpu.cpp            # CPU register math vectorizations
├── cuda/                          # GPU Optimization Stack
│   └── gguf_cuda.cu               # Ampere __dp4a dynamic quantization kernels
└── api/                           # Production Network Gateways
    └── api_server.py              # Anthropic-compatible ASGI Python server

```

## Build and Installation Pipeline

1. Execute System Compilation
   Run the automated environment setup tool. This script validates your system compilers (gcc, nvcc), prepares an isolated Python virtual environment shell, installs required ASGI gateway dependencies, and runs CMake to generate your target binary:

```bash
chmod +x setup.sh
./setup.sh
```

Alternatively, to build or modify individual target configurations manually via CMake, utilize standard terminal construction tracks:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc) 2. Interactive Terminal Generation
```

Launch the standalone engine directly by passing the destination to your local GGUF model along with your system memory allocation caps:

```bash
./build/moestream --model /nvme/glm-5.2-iq2_m.gguf --ram 48.0 --ctx 4096 3. Deploying the Anthropic-Compatible API
MoeStream includes a production-grade FastAPI wrapper that maps external HTTP network streams straight into your local executable's stdin/stdout pipe channels. It matches the native Anthropic /v1/messages format scheme, serving Server-Sent Events (SSE) token chunks asynchronously.
```

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

⚙️ Advanced 2-Bit Quantization (IQ2_M)
MoeStream includes dedicated lookup vectors built explicitly to unlock and execute heavily compressed 2-bit IQ2_M model variations.

Rather than running heavy tensor conversions across your PCIe boundaries, the engine reads packed index fields directly from your storage blocks, unpacks scale registers inside memory-aligned cache steps, and feeds your execution pipelines with minimal latency.
