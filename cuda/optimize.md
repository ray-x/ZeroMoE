Yes, those original suggestions absolutely still apply, but they have evolved. Because you added the **RTX 5060 Ti**, dropped the `zstd` compressed filesystem, and embraced **4-bit KV caching**, your engine's architecture transitions from a struggling single-GPU pipeline into an enterprise-grade, multi-GPU streaming engine.

The outdated "Pinned Double-Buffering" suggestion is now entirely superseded by **GPUDirect Storage (GDS)**, and the single-GPU cache expands into a specialized asymmetric dual-GPU cluster.

Here is the comprehensive, detailed software design document for optimizing your C++ inference engine on your specific hardware (Intel W2235, PCIe Gen 3, NVMe, RTX 3090 + RTX 5060 Ti).

---

## 1. System Memory Map (Asymmetric Dual-GPU Allocation)

To completely bypass your PCIe Gen 3.0 bottleneck, the engine must never move data unnecessarily. Every tensor must have a permanent home.

### GPU 0: RTX 5060 Ti 16GB (The Core Engine)

This card executes the baseline intelligence of the model on every single token.

* **Dense Layers (4-Bit):** ~7.2 GB. Token Embeddings, Multi-head Latent Attention (MLA) parameters, Layer Norms, Routers, and the LM Head.
* **KV Cache (4-Bit, 150k Context):** ~5.3 GB. The full engineering codebase history.
* **Activation State (`hidden_states`):** ~500 MB. Permanently allocated via `cudaMalloc` to prevent CPU-to-GPU bouncing.
* *Remaining Headroom:* ~3.0 GB for CUDA driver overhead and transient math scratchpads.

### GPU 1: RTX 3090 24GB (The Expert Vault)

This card acts as a massive, high-speed memory controller.

* **MoE Expert LRU Cache (2-Bit):** ~23.0 GB. Dedicated entirely to holding the uncompressed `UD-IQ2_M` feed-forward experts.
* *Remaining Headroom:* ~1.0 GB for the CUDA driver.

---

## 2. Multi-GPU Execution Pipeline (`execution_layer.cpp`)

Your forward pass will now act as a traffic controller, splitting the compute between the two distinct CUDA device IDs.

1. **Prefill & Attention (GPU 0):** The engine reads the token, executes the embeddings, unpacks the 4-bit KV cache on the fly, and computes the MLA attention scores entirely on the 5060 Ti.
2. **Routing (GPU 0):** The router projection determines the top-$K$ experts required for the current token and passes those IDs to the CPU orchestrator.
3. **The Expert Fetch (GPU 1):** The CPU checks the LRU cache map for the RTX 3090.
* *Cache Hit:* The expert is ready; skip to execution.
* *Cache Miss:* The engine triggers a GPUDirect `cuFileRead` to DMA the missing 2-bit expert directly from the uncompressed NVMe drive into the 3090's VRAM.


4. **Expert Compute (GPU 1):** The 3090 executes the `IQ2_M` dequantization and computes the Feed-Forward Network (FFN) math.
5. **Cross-Device Accumulation:** The tiny output vector (e.g., 4096 FP16 floats) from the 3090 is transferred over the PCIe bus back to the 5060 Ti to perform the residual addition. *(Note: If direct P2P is blocked by the driver due to mixed architectures, this bounces safely through a tiny pinned host buffer).*

---

## 3. GPUDirect Storage Implementation (`cuda_manager.hpp`)

You will rip out the POSIX `pread` and `LockFreePrefetcher` logic, replacing it with the `cuFile` API.

* **File Registration:** Open the uncompressed 250GB GGUF file using the `O_DIRECT` flag (to bypass the Linux OS page cache) and register it using `cuFileHandleRegister()`.
* **VRAM Registration:** Pre-allocate your 23GB expert cache on the RTX 3090 using `cudaMalloc` and register that memory block with the driver using `cuFileBufRegister()`.
* **DMA Reads:** When a cache miss occurs, calculate the exact byte offset of the expert from the GGUF header and call `cuFileRead()`. The NVMe controller will push the bytes straight to the 3090 without the CPU touching a single bit.

---

## 4. Custom CUDA Dequantization Kernels (`gguf_cuda.cu`)

To support the Mixed Precision architecture, you must write two distinct dequantization kernels to replace the CPU mock loops.

* **4-Bit KV / Dense Kernel (For GPU 0):** A kernel designed for the Blackwell architecture of the 5060 Ti. It reads the packed 4-bit `uint8_t` memory, shifts the nibbles, applies the FP16 scales, and feeds the native `__dp4a` dot-product matrix multiplication.
* **2-Bit `UD-IQ2_M` Kernel (For GPU 1):** A highly specialized kernel for the Ampere architecture of the 3090. It loads the 256-value `iq2_m_codebook` array into the GPU's ultra-fast `__shared__` memory. Warp threads will read the 2-bit bitstreams, use the shared memory codebook to resolve the floating-point values, and execute the math.

---

## 5. Loop and Memory Hardening

To eliminate the CPU-side heap churn that causes micro-stutters:

* **Pre-allocated Scratchpads:** Inside `moe_forward_gguf`, completely remove dynamic instantiations like `std::vector<ExpertScore> scores(n_experts)`. Move these arrays into a persistent `LayerScratch` struct allocated once during engine initialization.
* **Token-Loop Flattening:** Ensure that the `for (int s = 0; s < S; ++s)` token loop does not contain a single `new`, `malloc`, or `std::vector` resize operation.

---

This design minimizes the impact of your PCIe Gen 3 constraints, weaponizes the mixed-architecture strengths of your GPUs, and fully prepares the engine to host a massive GLM-5.2 coding context.


To make your engine robust enough to handle any hardware combination, you need to implement a dynamic **Hardware Topology Detector** and an **Auto-Scaling Memory Planner**.

Instead of hardcoding device IDs, your engine must probe the system at startup, analyze the VRAM and architecture of each available card, and map the tensors accordingly. Here is the complete fallback design to handle all three scenarios.

### 1. Hardware Probing & Role Assignment

During startup, modify `CudaManager::initialize()` to probe all attached devices using `cudaGetDeviceCount` and `cudaGetDeviceProperties`. The engine will assign roles based on the following logic:

* **The Core Engine (Dense + KV Cache):** The engine looks for the GPU with the newest architecture (e.g., Blackwell) or the lowest VRAM that is $\ge$ 16GB.
* **The Expert Vault (MoE Cache):** The engine assigns all remaining GPUs (highest VRAM capacity) to act as a unified, spanned LRU cache pool.

---

### 2. The Three Fallback Profiles

#### Profile A: Single GPU Fallback (1x RTX 3090 24GB)

If the engine detects only one GPU, it must cram the entire pipeline onto a single device. The priority is preventing an Out of Memory (OOM) crash, which means sacrificing the expert cache size.

* **Dense Layers (4-Bit):** ~7.2 GB allocated on GPU 0.
* **KV Cache (4-Bit):** Scaled down dynamically. A 128k context (~4.5 GB) leaves the cache too small. The engine automatically caps the context at **64k tokens (~2.2 GB)**.
* **MoE Expert Cache (2-Bit):** The remaining ~12.5 GB of VRAM is dedicated to the LRU cache.
* **Execution Flow:** Everything runs sequentially on GPU 0. GPUDirect Storage (GDS) via `cuFile` DMA streams missing experts directly into GPU 0's limited cache pool.

#### Profile B: Dual GPU Mode (1x RTX 5060 Ti 16GB + 1x RTX 3090 24GB)

The engine detects two GPUs and immediately splits the architecture to stop PCIe bottlenecking.

* **Core Engine (GPU 0 - 5060 Ti):**
* Dense Layers (4-Bit): ~7.2 GB.
* KV Cache (4-Bit): Restored to full **128k–150k context (~4.5 GB to 5.3 GB)**.


* **Expert Vault (GPU 1 - 3090):**
* The entire ~23 GB of usable VRAM is allocated as the MoE LRU cache.


* **Execution Flow:** GPU 0 computes attention and routing. It sends the active token hidden states to GPU 1 via Peer-to-Peer (P2P) PCIe transfer. GPU 1 pulls 2-bit experts via GDS, executes the math, and sends the output vector back to GPU 0 for the residual addition.

#### Profile C: Triple GPU Cluster (1x RTX 5060 Ti 16GB + 2x RTX 3090 24GB)

The engine detects three GPUs. The Core Engine remains identical to Profile B, but the MoE routing becomes distributed.

* **Core Engine (GPU 0 - 5060 Ti):** Dense Layers (~7.2 GB) + KV Cache (~5.3 GB).
* **Spanned Expert Vault (GPU 1 & GPU 2 - 3090s):** The LRU cache expands to **~46 GB**.
* **Execution Flow:** The CPU orchestrator maintains a unified hash map of which experts are on which card.
* If the router selects Expert $E_5$, and the map shows it resides on GPU 2, the engine dispatches the compute stream to GPU 2.
* If a cache miss occurs, the LRU manager evicts the oldest expert across *both* 3090s and targets the GDS DMA transfer to the card with the evicted slot.



---

### 3. Implementation in the Codebase

To make this seamless, you will introduce a `MemoryPlanner` inside `engine.hpp` that executes before the `resident_layers` are populated.

**1. Update `cuda_manager.hpp**`


You must upgrade your single-device manager to a pool manager.

```cpp
struct GpuDevice {
    int id;
    size_t total_vram;
    size_t available_vram;
    cudaStream_t compute_stream;
    cudaStream_t copy_stream;
};
std::vector<GpuDevice> available_gpus;

```

**2. Dynamic Allocation in `graph_initializer.cpp**`


Instead of blindly loading the dense manifest, the engine loops through the manifest and checks `available_gpus[0].available_vram`. If it fits, it calls `cudaMalloc` on GPU 0. If it overflows, it throws an exception advising the user to reduce the `--ctx` parameter.

**3. LRU Cache Redesign in `ram_manager.hpp**`


Your `RAMSafetyManager` must be rewritten to manage a `std::unordered_map<int, ExpertLocation>`:

```cpp
struct ExpertLocation {
    int gpu_id;
    uint8_t* device_ptr;
    uint64_t last_used_clock;
};

```

When `get_expert()` is called, it returns the `gpu_id` holding the weight. The execution loop in `src/execution_layer.cpp` then targets that specific device ID for the CUDA kernel launch.

Here is the explicit, hardcoded memory and layer mapping for your three specific hardware configurations. This dictates exactly which tensors and data structures from your engine live on which physical card.

### Configuration A: 1x RTX 3090 (24GB) Only

Since everything must fit on one card, the VRAM is strictly partitioned to prevent an Out of Memory (OOM) crash, leaving a smaller footprint for the Mixture-of-Experts (MoE) cache.

* **GPU 1 (RTX 3090 - 24GB):**
* **Dense Layers (4-Bit):** `token_embd.weight`, `output.weight`, and all MLA/router projections (`attn_q_a.weight`, `attn_output.weight`, `ffn_gate_inp.weight`). (~7.2 GB)


* **KV Cache (4-Bit):** The `engine.Lc` and `engine.Rc` historical context buffers. (~4.5 GB for 128k context)


* **Active State:** The `hidden_states` vector and temporary intermediate matrix computations. (~1.0 GB)


* **MoE Expert Cache (2-Bit):** Holds the frequently used `ffn_gate`, `ffn_up`, and `ffn_down` expert slab allocations. (~11.3 GB)




* **GPU 2:** N/A
* **GPU 3:** N/A

---

### Configuration B: 1x RTX 5060 Ti (16GB) + 1x RTX 3090 (24GB)

This isolates the static, every-token computational baseline on the Blackwell architecture (5060 Ti) while dedicating the entire Ampere architecture (3090) to MoE caching and GPUDirect DMA streaming.

* **GPU 1 (RTX 5060 Ti - 16GB):**
* **Dense Layers (4-Bit):** `token_embd.weight`, `output.weight`, and all MLA/router projections (`attn_q_a.weight`, `attn_output.weight`, `ffn_gate_inp.weight`). (~7.2 GB)


* **KV Cache (4-Bit):** The `engine.Lc` and `engine.Rc` historical context buffers. (~4.5 GB to 5.3 GB for 128k–150k context)


* **Active State:** The `hidden_states` vector and temporary intermediate matrix computations. (~1.0 GB)




* **GPU 2 (RTX 3090 - 24GB):**
* **MoE Expert Cache (2-Bit):** Dedicated 100% to holding the `ffn_gate`, `ffn_up`, and `ffn_down` expert slab allocations. (~23.0 GB)




* **GPU 3:** N/A

---

### Configuration C: 1x RTX 5060 Ti (16GB) + 2x RTX 3090 (24GB)

This is the ultimate setup for your coding assistant. The static computational baseline remains on the 5060 Ti, while the MoE LRU cache spans across both 3090s to hold nearly 46 GB of active coding experts.

* **GPU 1 (RTX 5060 Ti - 16GB):**
* **Dense Layers (4-Bit):** `token_embd.weight`, `output.weight`, and all MLA/router projections (`attn_q_a.weight`, `attn_output.weight`, `ffn_gate_inp.weight`). (~7.2 GB)


* **KV Cache (4-Bit):** The `engine.Lc` and `engine.Rc` historical context buffers. (~4.5 GB to 5.3 GB for 128k–150k context)


* **Active State:** The `hidden_states` vector and temporary intermediate matrix computations. (~1.0 GB)




* **GPU 2 (RTX 3090 - 24GB):**
* **MoE Expert Cache Split A (2-Bit):** Holds the first half of the most active `ffn_gate`, `ffn_up`, and `ffn_down` expert slabs. (~23.0 GB)




* **GPU 3 (External RTX 3090 - 24GB):**
* **MoE Expert Cache Split B (2-Bit):** Holds the second half of the most active `ffn_gate`, `ffn_up`, and `ffn_down` expert slabs. (~23.0 GB)