# CLAUDE.md

This file provides detailed architectural guidance, operational rules, hardware considerations, and historical pitfall documentation for working with the ZeroMoE codebase.

---

## 1. Overview and Architecture

ZeroMoE is a zero-dependency C++23 / CUDA17 inference engine built to stream Mixture-of-Experts (MoE) GGUF models (e.g., GLM-5.2, DeepSeek-V3, DeepSeek-V4 MLA, Hunyuan, Llama) on consumer hardware.

* **Core Paradigm:** Persistent dense layers are held in VRAM/RAM, while massive routed expert matrices are streamed on-demand from NVMe storage through a tiered cache.
* **MLA Absorbed Attention:** Folds value projection into query projections to compute attention scores in a low-rank latent space, reducing the operational footprint to $O(\text{Context} \times \text{kv\_lora})$.
* **RoPE Convention:** Uses interleaved YaRN scaling (pairs $2j, 2j+1$). **Do not use NeoX rotate-half**, as it scrambles positional encoding for supported MLA models.
* **Multi-GPU Context:** Configured for dual-GPU topologies (e.g., RTX 3090 "ExpertVault" + RTX 5060 Ti "CoreEngine") where the primary GPU handles expert caching and the secondary GPU handles dense layers alongside a dual-role cache slice.

---

## 2. Build & Execution Rules

### CUDA MoE onboarding: required stop/go workflow
CUDA MoE support requires a registered `FullGraphModel` adapter on the shared graph runtime. Do not onboard a family through the legacy layer-by-layer expert executor. Before coding, trace the exact GGUF architecture through `detect_model_family()`, `resolve_family_ops()`, and `full_graph_model()`. Record architecture/topology checks and the llama.cpp source path/commit.

Map each llama.cpp block stage to the ZeroMoE adapter before implementation, including tensor shape/layout, dtype, normalization function and epsilon, scale placement, projection/activation order, residual order, router behavior, and state handoff. Preserve the reference operation sequence. Do not use an algebraically equivalent operation or fuse/reorder work without a same-input boundary-trace A/B; small rounding changes can alter routed experts downstream.

Gate order is mandatory:
1. Implement and register an adapter that whitelists the exact GGUF architecture and validates tensor topology before large allocation.
2. Obtain a decode memory plan. `Fits` requires positive required/available bytes with required <= available and mandates graph decode. `Unknown` also requires graph decode. Only `Insufficient` with required > available and a logged reason may permit layered fallback; graph-only models cannot fall back. `Unsupported` fails startup and never means memory fallback.
3. On a full-fit run, pass `python3 tools/check_full_graph_moe_run.py <run.log> --full-fit`. For Qwen3.5/Qwen3.6 also require `--require-all-expert-layers-on-device --require-full-graph-prefill`. Graph build/run failure is fatal; never silently enter layered or legacy expert execution.
4. Teacher-force identical prompt and decode tokens through llama.cpp and ZeroMoE. Compare each layer's pre-MoE hidden states and routed expert IDs/weights; stateful models also compare recurrent state and prefill-to-decode handoff.
5. If values differ, freeze the GGUF, commits, device, context/positions, token IDs, threads, optional features, and graph policy. Find the first differing layer boundary, then instrument only that block. Check operation order against the llama.cpp source and compare stage-by-stage: input/norm, attention or GDN/state, residual, router logits/probabilities, top-k IDs/weights, per-expert gate/up/activation/down, shared expert, combine, and next hidden state. Align expert tensors by expert ID; record relative RMS, max absolute error, cosine similarity, and top-k margin. Fix the first cause and replay the same tokens. A route-ID mismatch or unexplained numerical drift keeps onboarding in progress.
6. Benchmark only after parity is restored and the full-fit graph gate passes. Match the same GGUF, physical GPU, context, active prompt length, generated token count, and thread count. Require at least 90% of llama.cpp by default and any user-set absolute floor via `--minimum-tps`. Keep the model in progress below either bar.

#### Debugging a numerical mismatch

Replay llama.cpp's token sequence with `ZEROMOE_TEACHER_FORCE_TOKEN_FILE` and `ZEROMOE_TEACHER_FORCE_SKIP_SAMPLING=1`; compare identical tensor boundaries, not free-running text. For Qwen, use `ZEROMOE_QWEN4EXP_PROBE_DUMP=<dir>` and `ZEROMOE_QWEN4EXP_PROBE_LAYER=<layer>` for graph/router and selected expert-stage taps. Inspect the earliest differing boundary, align expert slots by ID, and record numeric error plus the top-k margin. Fix and replay that boundary before tuning downstream layers. Never waive operation-order or routing differences as “mathematically equivalent” without a same-input A/B.

#### When onboarding is finished

Mark a model supported only for the scope actually tested: exact GGUF architecture/quantization, device, context, and input path. Require all of the following evidence:

- Exact architecture registration and topology validation pass.
- A valid live memory plan and the full-fit graph gate pass; no layered or legacy per-expert dispatch occurs when the plan fits.
- Teacher-forced pre-layer hidden states and routed IDs/weights match llama.cpp across layers; stateful models also compare state handoff.
- Repeated matched performance measurements pass the 90% llama.cpp bar and any user-set `--minimum-tps` target.
- Prefill, graph build/upload, steady decode, and end-to-end rates are reported separately, with source/toolchain/GGML provenance and persistent logs/results.

Passing compilation, coherent output, or only one of the graph/performance gates is not enough. An `Insufficient`-memory fallback may be documented as memory-limited support for that scope, but does not establish full-fit support; `Unknown` is never acceptance. State untested quantizations, devices, multimodal paths, and optional features instead of implying support for them. The full checklist is in `docs/design/full-graph-model-onboarding.md`.

### System Requirements & Constraints
* **Platform:** Linux, NVIDIA CUDA Toolkit, GCC (C++23 support), Python 3, and an AVX-512 capable CPU. 
* **Apple Silicon / Non-CUDA Limitations:** Will not build or run on macOS/Apple Silicon due to hard NVCC and x86 AVX-512 dependencies.

### Commands
* **Standard Build:** `./setup.sh` or `make` to configure and compile `./build/zeromoe`.
* **Testing:** `make test` executes the native test harness (`tests/native/main_test.cpp`). CPU-only components can be tested cross-platform via `make test-components` (`tests/test_components.cpp`).
* **Execution:** `./build/zeromoe --config configs/single_gpu.toml` (preferred over legacy CLI flags).
* **API Gateway:** `api/` currently holds only a `venv`; the Anthropic-compatible ASGI server script (`api_server.py`) is not in the tree.

### Source Layout
There is no `src/` or `include/` tree. Sources and headers live side by side in: `core/` (engine config, `tensor_trace.hpp`), `memory/` (`direct_io.hpp`, `io_uring_backend.hpp`, RAM/VRAM caches), `models/{base,graph,qwen,glm,mimo,deepseek}/`, `backends/{cpu,cuda,ggml,exl3}/`, `kernels/` (CUDA quant kernels), `tests/` (`tests/native/` is the native harness). `ZEROMOE_INCLUDE_DIRS` in `CMakeLists.txt` is the include-path list; `tests/run_components.sh` mirrors it.

### GPU Co-existence Guard
Before launching any run or benchmark, check for competing GPU processes using `nvidia-smi` and `pgrep -af` (e.g., `llama.cpp`, `llama-bench`, `ollama`, `vllm`). ZeroMoE computes its internal VRAM budgets (`moe_cache_gb`, active-layer pool, dual-role slice) once at startup based on `cudaMemGetInfo`. External processes consuming VRAM during or before startup induce hard-to-diagnose OOMs and cache thrashing.

### Non-CUDA / macOS Dev Iteration Workarounds
Building directly on non-CUDA hosts hard-fails at `find_package(CUDAToolkit REQUIRED)`. Instead of CMake, perform syntax checks using LLVM standalone tools:
* `/opt/homebrew/opt/llvm/bin/clang-check <file> -- -std=c++23 -Icore -Imemory -Imodels/base -Imodels/graph -ferror-limit=0`
  * *Crucial:* Always pass `-ferror-limit=0` to prevent early diagnostic truncation caused by missing CUDA headers.
* `/opt/homebrew/opt/llvm/bin/clang-tidy -checks='bugprone-*,performance-*' <file> -- -std=c++23 -Icore -Imemory -Imodels/base -Imodels/graph`
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
  * `"global_topn"`: Pools all expert routing counts from `.zeromoe_usage` into a single global ranking and pins the hottest experts across the entire model, subject to a minimum floor per layer (`pin_global_floor_per_layer`, default 1).
* **TOML Scoping Rule:** In configuration TOML files, place all root-level configuration keys **before** the first `[[gpus]]` table header. Keys placed after a `[[gpus]]` header are parsed as fields of that GPU entry.
* **CPU-Affinity Layers (`cpu_affinity_layers`):** Array of layer indices (e.g., `cpu_affinity_layers = [4, 5, 7]`) that permanently pins specified layers to CPU dispatch. Prevents VRAM uploads and preloading for layers where NVMe reads dwarf compute savings.

---

## 4. Execution Dispatch & Quantization

ZeroMoE utilizes a hybrid execution model, favoring the GPU but falling back to AVX-512 CPU execution for unsupported types.

* **GPU Fused Kernels:** Supported for `Q4_0`, `IQ2_M`, and `IQ4_XS`. Operates asynchronously on `copy_stream` to allow Pass A / Pass B overlapping.
* **CPU AVX-512 Fallback:** Optimized SIMD dequantization routines exist for `IQ2_XXS`, `IQ3_XXS`, and `IQ2_S`. Execution speed is bound by SIMD dequantization throughput.
* **Matched-Type BF16 Fast-Path:** Dense outputs like `attn_output` and `gate_inp` use a budget-checked fast path. Weights are dequantized host-side to `BF16` and executed using `cublasGemmEx` (BF16 weight $\times$ BF16 activation), reducing VRAM usage by 50% vs F32.
* **GPU Batched MLA Q-Absorb (`attn_qabsorb_gpu`):** Replaces CPU per-head attention loops with strided batched GEMMs (`cublas_dense_matmul_bf16_matched_batched`). `CudaManager::upload_wuk_bf16` pre-uploads WUK weights during warmup (`warm_mla_absorb_cache`).

---

## 5. Debugging & Tensor Tracing

`core/tensor_trace.hpp` provides unified VERBOSE tracing and NaN detection across model layers.

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
* **Nested OpenMP Regions:** Never call functions containing internal `#pragma omp parallel` loops from inside an outer parallel region. ZeroMoE runs with nested OMP disabled; nested parallel regions execute sequentially while paying thread-team spawn and barrier overhead.
* **Batch Size Parallelism Splitting:** Parallelize across heads/experts in single outer `#pragma omp parallel for` loops. Avoid spawning OMP parallel regions inside per-token loops for $S=1$ decode steps.

---

## 7. Commit Rules

* **No `Co-Authored-By` trailers** (and no "Generated with Claude Code" lines) in commit messages or PR descriptions. Commits carry the author's identity only.

---

## 8. Supported Models, Builds and Validation (as of 2026-10-23)

### Models and where they are documented
Full-graph model families: Qwen 3.8 Flash, GLM-5.3 / GLM-5.3 Flash, Ling 3.1 Flash, DeepSeek-V4.1 Flash, MiMo V2.6 Flash. Each has a recipe and frozen baseline in `docs/models/<model>.md`; measured numbers live in `configs/bench/perf-baselines.md` (update it on a gain, and tell the owner of any drop above 5%). The shared MoE/model-family framework is described in `docs/design/model-family-framework.md`; settings resolution (defaults, family overrides, environment) in `docs/design/settings-resolver.md`. CUDA MoE decode is full-graph only when the adapter planner reports Fits or Unknown. A graph failure aborts the request. Layered fallback is allowed only for an explicit Insufficient plan with required > available bytes and a logged reason.

Qwen3.5/Qwen3.6 GGUFs use the generic `Qwen` family and `qwen35moe`
architecture. An initial full-graph decode adapter now reuses the shared
Qwen GDN, full-attention, and routed-MoE operators while building the standard
Qwen norm/residual block. It plans all weights plus KV/GDN state and workspace,
and requires every expert layer to be device-resident on a full-fit plan.
Do not call this CUDA onboarding complete until a hardware run passes
`tools/check_full_graph_moe_run.py --full-fit
--require-all-expert-layers-on-device`; the adapter has not yet been
fully built or performance-validated. Ornith uses this same architecture, but its
previous run still needs a captured gate log for verification.

### Builds
* Always configure with `-DCMAKE_BUILD_TYPE=Release` (an unset build type compiles without `-O`).
* Builds that use the unmodified ggml series live in the main checkout: `build_master` (engine) and `build_master_ggml` (patched ggml). Builds that need branch-only ggml patches live in the worktree that carries the patches and may be deleted after it is merged. A worktree build links its ggml libraries by absolute path, so copying the binary out of it does not work; rebuild in the main checkout instead.
* The ggml patch recipe is `docs/build-ggml.md`, `scripts/build_ggml.sh`, `scripts/build_with_ggml.sh`. `patches/ggml`, `patches/ggml-host-node` and `patches/debug` are applied; `patches/experimental` and `patches/obsolete` never are.
* Benchmark only a binary with native CUDA kernels for the target GPU (for an RTX 5060 Ti, verify `sm_120a` with `cuobjdump --list-elf`); a build containing only `sm_86` is not a valid 5060 performance result. Separate first graph build/upload from steady decode timing.

### Defaults that affect VRAM and DDR
* `embedding_device` (`auto|cpu|gpu`) defaults to CPU/DDR for every model unless a config forces GPU. A graph that uploads its own resident tensors (MiMo) must skip `token_embd.weight` too.
* `moe_cache_reserve_gb` fits the expert cache to the free VRAM left after dense weights; an exhausted budget disables the cache, it does not mean "auto". Prefill-VRAM reuse is a common policy.
* Phase cache (`ZEROMOE_PHASE_CACHE_MIB`, default 0, reserve `ZEROMOE_PHASE_CACHE_RESERVE_MIB`): an optional, separately releasable decode-hot extension (`models/graph/phase_expert_cache.hpp`, design in `docs/design/phase-cache.md`). Validate with `bench/phase_cache_transition.py`. Validated on MiMo; on Ling it costs about 2.5-3% decode and is not enabled by default.

### Validating a model or an FFN change
* Compare decode logits with llama.cpp: README section "Validating a new model, or a change to a model's FFN flow" and `scripts/validate_vs_llama.sh`. Teacher-forced, per-frame comparisons are required; a first-step match proves nothing about decode.
* Teacher forcing through `bench/ling/reference-audit/gate.sh`/`run.sh` needs the token file in `LING_TOKEN_FILE` (the script unsets `ZEROMOE_TEACHER_FORCE_TOKEN_FILE`); `bench/phase_cache_transition.py` forces tokens per request and compares every decode frame. Never compare two free-running runs: one expert flip changes the text and every later measurement (the apparent 22% phase-cache slowdown on Ling came from two free-running runs whose routes diverged at the first decode step; with tokens forced per request the cost was about 2.5-3%).
* Kernel-level check, free of model noise: `scripts/check_kernel_numerics.sh` (aliased expert views bitwise equal to contiguous tensors; CPU and CUDA within 1e-4 of float64 on exactly representable inputs).
* Per-model gates: MiMo keeps cosine >= 0.999 against its control placement. Ling is gated by the kernel check plus a bitwise-identical repeated request instead (decision of 2026-10-23), because moving Ling's experts between CPU and GPU changes logits by cosine 0.95-0.99 even in stock llama.cpp: the CPU rounds activations to 8 bits in 256-value blocks (Q8_K) and CUDA in 32-value blocks (Q8_1), and Ling's spiky swiglu output makes the CPU error about twice as large. The kernels themselves are exact. Details and a rejected fix (+7-8% on the CPU expert op) are in `docs/experiments/ggml-cpu-activation-rounding.md`; CPU expert-path performance is more important than placement-invariant numerics.
* `run.sh` refuses to start when `MemAvailable` is below `host_ram_gb + 21.5 GiB`. After a big run, drop the model files from the page cache with `posix_fadvise(POSIX_FADV_DONTNEED)` instead of root's `drop_caches`.

### Git workflow
* Work in a worktree, not in the main checkout. Master's history was rewritten during the source refactor, so `git rev-list master..branch` counts are meaningless for old branches; judge "merged" with `git cherry master <branch>` (patch content) and by checking that the code exists in master under the new paths.
* The project was renamed MoEStream -> ZeroMoE: the engine binary is `zeromoe`, env vars are `ZEROMOE_*`, usage files are `.zeromoe_usage*` (gitignored). Do not add new `moestream`/`MOESTREAM_*` names or absolute `/home/ray/Projects/MoEStream` paths. Remaining legacy references are deliberate: `patches/obsolete`, dated `docs/` handoffs, `glm_runs/`, `logs/`, and `bench/`/`configs/`/`tests/` scripts that point at real external directories still named `moestream-*` on disk (e.g. `~/Projects/moestream-build-upstream-master`, `~/work/moestream-graph`, `/data/moestream-offload`). Rename those references only together with the directories.
