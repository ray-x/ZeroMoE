# ggml patch series

MoEStream builds against **upstream ggml-org/llama.cpp** plus the patches here;
it does not carry a modified copy of ggml.

- Upstream base: `3cf03257f` ("CUDA: enable sparse fa for qwen4 (#28770)").
- Patch paths are relative to the `ggml/` directory (`a/src/ggml-cuda/...`).
- Applied automatically at configure time by `cmake/MoEStreamGgmlPatches.cmake`
  (idempotent: a patch that already applies in reverse is skipped).
- Manual: `cd <llama.cpp>/ggml && for p in <repo>/patches/ggml/*.patch; do patch -p1 < $p; done`

## Layout

| Directory | Applied by CMake | Purpose |
|---|---|---|
| `ggml/` | always (required) | Core series |
| `ggml-host-node/` | `MOESTREAM_ENABLE_GGML_HOST_NODE_PATCHES` (default ON), required | CUDA-graph host-node bridge for the Qwen4Exp CPU/GPU split |
| `ggml-host-node-beellama/` | same option, best effort | Needs beellama-only scheduler profiling helpers (`ggml_backend_sched_profile_is_cpu`, ...); **does not apply to plain upstream** — see "Pending port" |
| `debug/` | `MOESTREAM_GGML_DEBUG_PATCHES` (default OFF) | Profiling / logging / bench instrumentation. Not part of the main series |
| `experimental/` | never | Spikes and A/B experiments |
| `obsolete/` | never | Superseded or stale; kept for reference |

## Core series (`ggml/`)

| Patch | Merged from | Files |
|---|---|---|
| `0001-ggml-cpu-q2_0-vnni` | q2-vnni, q2-vnni-multirow, q2-vnni-mul-mat-id-batch | `arch-fallback.h`, `arch/x86/quants.c`, `ggml-cpu.c` |
| `0002-ggml-mul-mat-id-negative-skip` | (unchanged) | `ggml-cpu.c`, `ggml-cuda/mmvq.cu` |
| `0003-ggml-cuda-graph-capture` | cuda-stream-bridge, cuda-conditional-graph-harvest, cuda-graph-first-call-capture | `ggml-cuda/ggml-cuda.cu` |
| `0004-ggml-cuda-external-pointer-padding` | (unchanged) | `ggml-cuda.cu`, `mmq.cu`, `mmvq.cu` |
| `0005-ggml-backend-static-weight-copy` | backend-static-weight-copy-count | `ggml-backend.cpp` |
| `0011-ggml-cuda-mestream-bridge-compat` | bridge-only compatibility hunk for newer upstream context | `ggml-cuda/ggml-cuda.cu` |
| `0012-ggml-impl-export-graph-view` | export graph view from shared ggml | `ggml-impl.h` |

`ggml-host-node/0001-ggml-cuda-host-node` = cuda-host-node-bridge + cuda-host-node-parallel.
`ggml-host-node/0002-ggml-cuda-host-node-parallel-gate` = the prefill-safe gate, re-derived from the integration fork's
`ggml-cuda.cu` (the original patch targeted beellama's text and did not apply to upstream).

Originals of every merged patch are in `obsolete/superseded-by-merge/`.

### Verification of the merge (2026-10-05)

Applying the original 11 patches in sequence and applying the merged series to a
clean checkout of the base give **byte-identical trees** (git tree
`fb8762929ed58d27e41751cc4669b501821b47ae`). A configure against a pristine
upstream checkout applies all seven, and a second configure applies none.
With the gate patch added, the resulting tree is `9b04e2d634ff2645caf8ef08dfd4ffa60c6530fc`; a full
`moestream` build against it succeeds (0 errors).

## Patches that did not make the main series

- **Pending port** (`ggml-host-node-beellama/`): `backend-host-node-gate`,
  `backend-host-node-coordination`, `backend-host-node-coordination-enable`. They patch beellama's scheduler
  profiling code, which upstream lacks. To use them on upstream, re-express the
  `ggml_backend_sched_profile_is_cpu(...)` checks as `ggml_backend_is_cpu(...)`
  and rebase. Until then, a plain-upstream build runs without host-node
  scheduler coordination.
- `debug/ggml-cpu-moe-node-profile` — `MOESTREAM_QWEN4EXP_CPU_CALLBACK_BREAKDOWN`
  profiling; needs the same beellama base.
- `debug/bee-bench-exact-decode-and-phase-timing` — patches beellama's
  `llama-bench` (exact-decode token files, phase timing). For the comparison
  benchmark, not for MoEStream.
- `experimental/ggml-bf16-mmvf-cublas-experiment` — env-gated BF16 matvec via cuBLAS.
- `experimental/ggml-scheduler-early-ready-layer13` — Layer-13 predictive staging spike.
- `obsolete/ggml-cuda-cccl-iterators` — upstream now includes `<cuda/iterator>`
  inside its own CCCL guard; the CUDA-11 guard it added is irrelevant on CUDA 13.
- `obsolete/exl3-runtime-cmake`, `obsolete/exl3-cuda-runtime-cmake` — edits to
  MoEStream's own old `CMakeLists.txt`; no longer apply (the sources they added
  are in the CMake file already).
