# CLAUDE.md

This file provides detailed architectural guidance, operational rules, hardware considerations, and historical pitfall documentation for working with the MoEStream codebase.

---

## 1. Overview and Architecture

MoEStream is a zero-dependency C++23 / CUDA17 inference engine built to stream Mixture-of-Experts (MoE) GGUF models (e.g., GLM-5.2, DeepSeek-V3, DeepSeek-V4 MLA, Hunyuan, Llama) on consumer hardware.

* **Core Paradigm:** Persistent dense layers are held in VRAM/RAM, while massive routed expert matrices are streamed on-demand from NVMe storage through a tiered cache.
* **MLA Absorbed Attention:** Folds value projection into query projections to compute attention scores in a low-rank latent space, reducing the operational footprint to $O(\text{Context} \times \text{kv\_lora})$.
* **RoPE Convention:** Uses interleaved YaRN scaling (pairs $2j, 2j+1$). **Do not use NeoX rotate-half**, as it scrambles positional encoding for supported MLA models.
* **Multi-GPU Context:** Configured for dual-GPU topologies (e.g., RTX 3090 "ExpertVault" + RTX 5060 Ti "CoreEngine") where the primary GPU handles expert caching and the secondary GPU handles dense layers alongside a dual-role cache slice.

---

## 2. Build & Execution Rules

### System Requirements & Constraints
* **Platform:** Linux, NVIDIA CUDA Toolkit, GCC (C++23 support), Python 3, and an AVX-512 capable CPU. 
* **Apple Silicon / Non-CUDA Limitations:** Will not build or run on macOS/Apple Silicon due to hard NVCC and x86 AVX-512 dependencies.

### Commands
* **Standard Build:** `./setup.sh` or `make` to configure and compile `./build/moestream`.
* **Testing:** `make test` executes the native test harness (`src/main_test.cpp`). CPU-only components can be tested cross-platform via `make test-components` (`tests/test_components.cpp`).
* **Execution:** `./build/moestream --config configs/single_gpu.toml` (preferred over legacy CLI flags).
* **API Gateway:** `source api/venv/bin/activate && python3 api/api_server.py` to launch the Anthropic-compatible ASGI Python server.

### GPU Co-existence Guard
Before launching any run or benchmark, check for competing GPU processes using `nvidia-smi` and `pgrep -af` (e.g., `llama.cpp`, `llama-bench`, `ollama`, `vllm`). MoEStream computes its internal VRAM budgets (`moe_cache_gb`, active-layer pool, dual-role slice) once at startup based on `cudaMemGetInfo`. External processes consuming VRAM during or before startup induce hard-to-diagnose OOMs and cache thrashing.

### Non-CUDA / macOS Dev Iteration Workarounds
Building directly on non-CUDA hosts hard-fails at `find_package(CUDAToolkit REQUIRED)`. Instead of CMake, perform syntax checks using LLVM standalone tools:
* `/opt/homebrew/opt/llvm/bin/clang-check <file> -- -std=c++23 -Iinclude -ferror-limit=0`
  * *Crucial:* Always pass `-ferror-limit=0` to prevent early diagnostic truncation caused by missing CUDA headers.
* `/opt/homebrew/opt/llvm/bin/clang-tidy -checks='bugprone-*,performance-*' <file> -- -std=c++23 -Iinclude`
* Disregard expected `cuda_runtime.h file not found` and x86 `immintrin.h` errors on ARM hosts; filter diagnostics to code changes. Full compilation must still be verified on the Linux/CUDA host.

---

## 3. Tiered Expert Caching & Memory Architecture

Routed experts are loaded per-token through a strict caching hierarchy:

```
[NVMe Direct I/O] ---> [Host RAM LRU] ---> [Active Layer Area (VRAM)] ---> [GPU VRAM LRU Cache]
```

### Cache Tiers
1. **GPU VRAM Cache:** Managed by `GpuTopology` on ExpertVault GPUs. Uses permanent pinning and a temporary `hold_for_compute()` mechanism to prevent concurrent eviction races during kernel execution.
2. **Active Layer Area (Three-Tier Buffer):** Dedicated VRAM buffer (`active_layer_capacity`) holding current decode-time cold-miss experts. Experts crossing `active_layer_promote_hit_threshold` are promoted directly via device-to-device copy (`promote_active_to_main`) to the main VRAM LRU cache. Unused cold entries are freed at layer transitions via `cleanup_active_layer_area`.
3. **Host RAM LRU:** Managed by `RAMSafetyManager`, utilizing frequency + recency eviction. Capped safely to prevent OS swapping. Uses an atomic `claim_or_wait` design to unify prefetch and decode requests.
4. **Disk Prefetching:** Lock-free SPSC ring buffer predicts next-layer experts via a pilot router pass. Reads use `O_DIRECT` (`direct_io.hpp`) with 4096-byte alignment. Opt-in `io_uring` (`io_uring_backend.hpp`, `io_uring=true`) batches gate/up/down reads into unified syscall submissions.

### Key Tuning & Configuration Options
* **Pin Strategy (`pin_strategy`):**
  * `"per_layer_topk"` (default): Assigns a fixed pin count (`pin_topk_per_layer`) to every layer regardless of routing concentration.
  * `"global_topn"`: Pools all expert routing counts from `.moestream_usage` into a single global ranking and pins the hottest experts across the entire model, subject to a minimum floor per layer (`pin_global_floor_per_layer`, default 1).
* **TOML Scoping Rule:** In configuration TOML files, place all root-level configuration keys **before** the first `[[gpus]]` table header. Keys placed after a `[[gpus]]` header are parsed as fields of that GPU entry.
* **CPU-Affinity Layers (`cpu_affinity_layers`):** Array of layer indices (e.g., `cpu_affinity_layers = [4, 5, 7]`) that permanently pins specified layers to CPU dispatch. Prevents VRAM uploads and preloading for layers where NVMe reads dwarf compute savings.

---

## 4. Execution Dispatch & Quantization

MoEStream utilizes a hybrid execution model, favoring the GPU but falling back to AVX-512 CPU execution for unsupported types.

* **GPU Fused Kernels:** Supported for `Q4_0`, `IQ2_M`, and `IQ4_XS`. Operates asynchronously on `copy_stream` to allow Pass A / Pass B overlapping.
* **CPU AVX-512 Fallback:** Optimized SIMD dequantization routines exist for `IQ2_XXS`, `IQ3_XXS`, and `IQ2_S`. Execution speed is bound by SIMD dequantization throughput.
* **Matched-Type BF16 Fast-Path:** Dense outputs like `attn_output` and `gate_inp` use a budget-checked fast path. Weights are dequantized host-side to `BF16` and executed using `cublasGemmEx` (BF16 weight $\times$ BF16 activation), reducing VRAM usage by 50% vs F32.
* **GPU Batched MLA Q-Absorb (`attn_qabsorb_gpu`):** Replaces CPU per-head attention loops with strided batched GEMMs (`cublas_dense_matmul_bf16_matched_batched`). `CudaManager::upload_wuk_bf16` pre-uploads WUK weights during warmup (`warm_mla_absorb_cache`).

---

## 5. Debugging & Tensor Tracing

`include/tensor_trace.hpp` provides unified VERBOSE tracing and NaN detection across model layers.

### Usage & Setup
* **Enable:** Set `log_level = "VERBOSE"` in the TOML configuration.
* **Filtering (`trace_filter`):** Specify tag prefixes (e.g., `trace_filter = "HC,MLA"`). Matches as anchored regex.
* **Abort on NaN (`trace_abort_on_nan = true`):** Immediately invokes `std::abort()` upon detecting non-finite values (NaN/Inf).

### Tag Categories
* `MLA` (Attention checks)
* `HC` (Hyper-Connections state)
* `HIDDEN` (Per-stage hidden-state buffers)
* `EXPERT-GPU` (GPU FFN buffers)
* `SILU-TRACE-GPU` / `SWIGLU-TRACE-GPU` (Activation states)
* `TRACE-ROUTER` (Router selections and weights)

### Instrumentation Idiom
```cpp
if (trace_gate("MY-TAG")) {
    trace_row("MY-TAG", layer_idx, "label", data_ptr, size, row_width);
}
trace_abort_if_nan("MY-TAG", contains_nan);
```

---

## 6. Critical Invariants, Findings, and Pitfalls

### Hyper-Connections & Model Initialization
* **Initialization Order Guard:** Hyperparameter reads (`c.hc_mult`, `c.hc_sinkhorn_iters`, `c.hc_eps`) in `graph_initializer.cpp` **must occur prior** to `resolve_family_ops()`. If `resolve_family_ops()` reads unitialized defaults, per-layer HC hooks (`hc_pre_fn`, `hc_post_fn`) silently default to no-ops. This causes context-blind generation collapse where outputs degrade into repetitive text.
* **Sinkhorn Normalization Structure:** `sinkhorn_normalize_comb` must execute: (1) Softmax over destination $+$ $\epsilon$, (2) One unconditional initial row-normalization step, and (3) $N-1$ alternating column-then-row normalizations.

### Multi-GPU & Constant Memory
* **Per-Device Constant Memory Initialization:** CUDA `__constant__` memory is allocated per-device. Constant table uploads (such as grid/sign tables in `iq2_matmul.cu`, `iq3_matmul.cu`, `iq4xs_matmul.cu`) must use per-device initialization flags (`ready[MAX_CUDA_DEVICES]`) indexed by `cudaGetDevice()`. Process-wide booleans leave non-primary GPUs with zeroed constant tables, resulting in zeroed output gates.

### Memory Management & Host Pinning
* **Async Transfer Memory Safety:** When preserving host RAM copies during promotion (`move_on_promotion=false`), `slot.pending_copy_event` must be set immediately in the promotion path. Unprotected RAM slots risk victim-selection scans reclaiming or reallocating `slab_buffer` while async H2D DMA is active.
* **Host Memory Registration Overhead:** Calling `cudaHostRegister` dynamically on short-lived host buffers during runtime adds high per-call driver overhead. For persistent scratch buffers (e.g., `LayerScratch`), allocate directly via `cudaMallocHost` (`PinnedFloatBuffer`) rather than registering `std::vector` allocations on the fly.

### Parallelism & Threading
* **Nested OpenMP Regions:** Never call functions containing internal `#pragma omp parallel` loops from inside an outer parallel region. MoEStream runs with nested OMP disabled; nested parallel regions execute sequentially while paying thread-team spawn and barrier overhead.
* **Batch Size Parallelism Splitting:** Parallelize across heads/experts in single outer `#pragma omp parallel for` loops. Avoid spawning OMP parallel regions inside per-token loops for $S=1$ decode steps.

---

## 7. Commit Rules

* **No `Co-Authored-By` trailers** (and no "Generated with Claude Code" lines) in commit messages or PR descriptions. Commits carry the author's identity only.
