# Copilot Instructions

## Build and tests

- Linux full-engine build (CUDA, C++23, and ggml):
  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DZEROMOE_USE_GGML=ON
  cmake --build build -j"$(nproc)"
  make test
  ```
- CPU-only build:
  ```bash
  cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DZEROMOE_CPU_ONLY=ON -DZEROMOE_USE_GGML=ON
  cmake --build build-cpu -j"$(nproc)"
  ./build-cpu/test_components
  ```
- Fast CPU component suite: `make test-components`.
- Run one component test by substring, for example: `make test-components ARGS="rope"`. Multiple filters are ORed: `make test-components ARGS="rope dequant"`.
- `make test` runs the native CUDA suite on a full build, or the component suite when only the CPU test binary is available. The CPU-only component script also runs its independent routing/cache checks.

Configure performance-sensitive builds as `Release`; an unset CMake build type is unoptimized.

## Architecture

ZeroMoE is a C++23/CUDA inference engine that reads GGUF metadata and streams routed MoE expert weights from NVMe while keeping persistent dense weights in host RAM or GPU memory. Expert access flows through disk direct I/O, a host-RAM LRU, a per-layer active VRAM area, and the GPU VRAM cache. Asynchronous prefetch uses an SPSC queue to prepare experts for upcoming layers.

Sources and headers are colocated: `core/` owns configuration, scheduling, and execution; `memory/` owns I/O and cache tiers; `models/graph/` provides the shared full-graph path and model-family builders live alongside it in `models/{base,qwen,glm,mimo,deepseek}/`; `backends/` implements CPU, CUDA, ggml, and EXL3 execution; `kernels/` contains CUDA kernels. `tests/native/` is the CUDA-capable harness; `tests/test_components.cpp` and related focused tests cover CPU-testable logic. Build options and include paths are defined in `CMakeLists.txt`.

### Full-model and CUDA-graph path

“Full graph” here means the whole-model ggml computation graph; CUDA graph capture/replay is performed by ggml's CUDA backend for its CUDA work. `models/graph/full_graph_model.hpp` is the engine-facing family adapter, `FullGraphModelBase` shares request validation and prefill/decode dispatch, and `graph_core.hpp` owns shared ggml handles, dense/expert/state arenas, and host inputs. Family graph implementations add architecture-specific attention and state while reusing the shared routing/MoE builders and execution strategies.

Keep graph construction and runtime policy separate from replay: the graph fixes tensor topology and bindings, while each replay supplies current inputs, route IDs/weights, and buffer contents. CUDA graph capture retains tensor addresses, so do not retarget captured weight pointers to new placements; keep borrowed buffers alive through pending work and retire/rebuild the scheduler when bindings or placement generations change. GGML owns capture at its backend boundary—do not wrap the whole multi-backend scheduler call in an external CUDA capture.

For the execution lifecycle and adding a family, start with `docs/design/full-graph-model-onboarding.md`; for the capture boundary and CUDA constraints, see `docs/cuda-graph-api-review-2026-10-05.md`. `ZEROMOE_USE_GGML_CUDA` enables the ggml CUDA backend, and `ZEROMOE_ENABLE_CUDA_GRAPHS` controls ggml CUDA graph capture/replay.

## Repository-specific conventions

- Attention uses interleaved YaRN RoPE pairs `(2j, 2j+1)`; do not substitute NeoX rotate-half.
- In TOML configs, put root-level settings before the first `[[gpus]]` table. Following keys belong to that GPU table.
- Read and initialize graph hyperparameters before resolving model-family operations: family hooks can otherwise capture defaults and silently become no-ops.
- CUDA constant memory and its initialization state are per device. Keep readiness flags indexed by the active CUDA device.
- Preserve host-buffer lifetime through asynchronous GPU transfers. When a source buffer remains in RAM during promotion, record its pending copy event before it can be reclaimed; use pinned allocation for persistent transfer scratch rather than repeatedly registering transient buffers.
- Avoid nesting code that creates OpenMP parallel regions inside another parallel region; the project runs with nested OpenMP disabled.
- Model and FFN correctness changes need teacher-forced, per-decode-frame comparisons against the reference, not just a first-token or free-running comparison. Use `scripts/validate_vs_llama.sh` where applicable and the kernel numerical check for kernel changes.
- Full-graph model requests fail if graph construction fails; there is no layered execution fallback.
