# ggml patch series

ZeroMoE builds against **upstream ggml-org/llama.cpp** plus the patches here;
it does not carry a modified copy of ggml.

- Upstream base: `8baabd53b77d1c48c6ef8d3b620436292f3f6231`.
- Patch paths are relative to the `ggml/` directory (`a/src/ggml-cuda/...`).
- Applied automatically at configure time by `cmake/ZeroMoEGgmlPatches.cmake`
  (idempotent: a patch that already applies in reverse is skipped).
- Manual: `cd <llama.cpp>/ggml && for p in <repo>/patches/ggml/*.patch; do patch -p1 < $p; done`

## Layout

| Directory | Applied by CMake | Purpose |
|---|---|---|
| `ggml/` | always (required) | Core series |
| `ggml-host-node/` | `ZEROMOE_ENABLE_GGML_HOST_NODE_PATCHES` (default ON), required | CUDA-graph host-node bridge for the Qwen4Exp CPU/GPU split |
| `ggml-host-node-beellama/` | same option, best effort | Needs beellama-only scheduler profiling helpers (`ggml_backend_sched_profile_is_cpu`, ...); **does not apply to plain upstream** — see "Pending port" |
| `debug/` | `ZEROMOE_GGML_DEBUG_PATCHES` (default OFF) | Profiling / logging / bench instrumentation. Not part of the main series |
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
| `0009-ggml-cpu-iq2-xxs-avx512bwvl` | `36f489d6b` AVX-512BW/VL IQ2_XXS dot kernel | `arch/x86/quants.c` |
| `0013-ggml-cpu-iq1-s-avx512vnni` | AVX-512 VNNI IQ1_S × Q8_K dot kernel | `arch/x86/quants.c` |
| `0014-ggml-build-commit-provenance` | preserve the pinned GGML base revision in copied build trees without `.git` | `CMakeLists.txt` |
| `0011-ggml-cuda-mestream-bridge-compat` | bridge-only compatibility hunk for newer upstream context | `ggml-cuda/ggml-cuda.cu` |
| `0012-ggml-impl-export-graph-view` | export graph view from shared ggml | `ggml-impl.h` |

`ggml-host-node/0001-ggml-cuda-host-node` = cuda-host-node-bridge + cuda-host-node-parallel.
`ggml-host-node/0002-ggml-cuda-host-node-parallel-gate` = the prefill-safe gate, re-derived from the integration fork's
`ggml-cuda.cu` (the original patch targeted beellama's text and did not apply to upstream).

Originals of every merged patch are in `obsolete/superseded-by-merge/`.

### Historical patch merge verification (2026-10-05; base `3cf03257f`)

Applying the original 11 patches in sequence and applying the merged series to a
clean checkout of the base give **byte-identical trees** (git tree
`fb8762929ed58d27e41751cc4669b501821b47ae`). A configure against a pristine
upstream checkout applies all seven, and a second configure applies none.
With the gate patch added, the resulting tree is `9b04e2d634ff2645caf8ef08dfd4ffa60c6530fc`; a full
`zeromoe` build against it succeeds (0 errors).

### GGML build revision metadata (2026-10-27)

The GGML CMake project normally discovers `GGML_COMMIT` by walking up from its
source directory. A copied build tree without `.git` could therefore inherit
ZeroMoE's commit and mislabel a performance baseline. Patch `0014` preserves
the revision supplied by ZeroMoE CMake. A configure against a temporary copy
of the pinned `8baabd53b77d1c48c6ef8d3b620436292f3f6231` source, with `.git`
removed, generated `ggml-version.h` containing that GGML revision.

## Patches that did not make the main series

- **Pending port** (`ggml-host-node-beellama/`): `backend-host-node-gate`,
  `backend-host-node-coordination`, `backend-host-node-coordination-enable`. They patch beellama's scheduler
  profiling code, which upstream lacks. To use them on upstream, re-express the
  `ggml_backend_sched_profile_is_cpu(...)` checks as `ggml_backend_is_cpu(...)`
  and rebase. Until then, a plain-upstream build runs without host-node
  scheduler coordination.
- `debug/ggml-cpu-moe-node-profile` — `ZEROMOE_QWEN4EXP_CPU_CALLBACK_BREAKDOWN`
  profiling; needs the same beellama base.
- `debug/bee-bench-exact-decode-and-phase-timing` — patches beellama's
  `llama-bench` (exact-decode token files, phase timing). For the comparison
  benchmark, not for ZeroMoE.
- `experimental/ggml-bf16-mmvf-cublas-experiment` — env-gated BF16 matvec via cuBLAS.
- `experimental/ggml-scheduler-early-ready-layer13` — Layer-13 predictive staging spike.
- `experimental/ggml-cpu-act32-*.diff` — CPU activation-rounding experiment (32-value `Q8_0` blocks for IQ-quant expert dot products). Rejected on performance (+7-8% on the CPU expert op); see `docs/experiments/ggml-cpu-activation-rounding.md`. Not applied by any build.
- `obsolete/ggml-cuda-cccl-iterators` — upstream now includes `<cuda/iterator>`
  inside its own CCCL guard; the CUDA-11 guard it added is irrelevant on CUDA 13.
- `obsolete/exl3-runtime-cmake`, `obsolete/exl3-cuda-runtime-cmake` — edits to
  ZeroMoE's own old `CMakeLists.txt`; no longer apply (the sources they added
  are in the CMake file already).
