# Building ZeroMoE against a patched ggml

Status 2026-10-23: the patch series is **ported to upstream llama.cpp `8baabd53b` (ggml 0.27.0)** and applies
with plain `git apply` (no fuzz). The ggml that the earlier Ling and MiMo gates ran against was a prebuilt
directory that was deleted by accident, and how it was patched had not been written down; this file and
`scripts/build_ggml.sh` are now the record.

## Rules

1. **A ZeroMoE binary must not depend on a directory it does not own.** The binary records the ggml
   directory as its RUNPATH. If that directory disappears, the loader silently falls back to the system
   `/usr/lib/libggml*` (sm_86 only, no PTX), and any kernel on the sm_120 GPU (RTX 5060 Ti) fails with
   `CUDA error: no kernel image is available ... ggml_cuda_kernel_can_use_pdl`. A running process keeps
   working (its libraries are already mapped), which hides the problem until the next start.
2. **Where builds live:**
   - a build of `master` (master's ggml patch series) lives inside the main checkout (`build_master`,
     `build_ggmlcuda`);
   - a build that needs branch-only ggml patches lives in that branch's worktree, and may be deleted
     when the worktree is merged back and removed.
3. `scripts/build_with_ggml.sh <build-dir>` refuses a `GGML_PREBUILT` that sits inside *another* worktree.
4. The way a ggml tree is patched must be a script or a document in this repo, never a one-off command.

## The series

`patches/ggml` then `patches/ggml-host-node`, in filename order, rooted at the ggml directory
(`a/src/...`), applied by `scripts/build_ggml.sh` (and by `cmake/ZeroMoEGgmlPatches.cmake` when CMake
builds ggml itself) with `git apply --directory=ggml`. A patch that does not apply stops the build.

Ported to 0.27 (same intent, rebased hunks): 0001 (Q2_0 VNNI; its `arch-fallback.h` hunk is gone because
upstream no longer defines the generic alias on x86), 0002 (negative-id skip; the new
`mul_mat_vec_q_moe` kernel needed the check added by hand), 0003 (CUDA graph capture on first call plus the
stream/graph bridges), 0004 (padding clear bounded by the buffer size; upstream now uses
`cudaMemsetAsync`), 0010 (f32 quant hook in `ggml_cuda_mul_mat`), 0012 (`GGML_API` on
`ggml_graph_view`). 0005, 0006, 0009, 0013 and the host-node series applied as written.
`patches/ggml-host-node/0003` (early-ready private CPU, the
`ggml_backend_sched_early_cpu_supported` symbol) was copied from the Ling worktree.

Moved to `patches/obsolete` (they target the older upstream): 0007 and 0008 (they patch the scheduler's
built-in MUL_MAT_ID expert copy, which upstream replaced with `ggml_backend_sched_set_copy_callback`;
ZeroMoE's `ExpertCopy` uses that callback; 0008's `ZEROMOE_SCHED_EVENTS` has no equivalent yet, MiMo's
op-offloaded prefill sets it, effect unmeasured), 0011 (bridge-only variant of 0003 for newer context;
0003 now carries the bridge) and the `0007-sched-early-cpu-hook` text patch (superseded by host-node/0003).

Verified: every `ggml_*`, `gguf_*` and `zeromoe_*` symbol the branch's binary imports is present in the
patched sources.

## Choosing a ggml source tree

Build from a clean upstream llama.cpp checkout at the pinned commit. Do not reuse
an unrelated or previously patched ggml tree unless its revision and patch state
have been verified.

### Getting the sources: `scripts/init_ggml.sh`

The pinned commit is the gitlink for `third_party/llama.cpp`. Fetch it with

```sh
scripts/init_ggml.sh
```

and not with `git submodule update --init`. The build reads only the `ggml/` directory (23 MB), but a plain submodule
checkout downloads the whole llama.cpp tree at that commit: about 390 MB of git objects and a 180 MB working tree, which
every fresh clone and every CI job would pay. The script makes a shallow, blob-less, sparse checkout instead: about 5 MB
in `.git`, 26 MB checked out, a few seconds. Its `ggml/` has the same git tree hash as the full checkout
(`090b01e56f56`), so the build is the same. CMake does the same slim fetch into the build tree if the submodule is
absent.

The build copies `ggml/` into the build directory and patches the copy, so `third_party/llama.cpp` stays unmodified.
If `git -C third_party/llama.cpp status` shows changes, someone patched the checkout in place; builds from that
checkout are not the pinned ggml.

## Verified (2026-10-23, binary `build_mimo` linked to `build_ggml`)

- Ling short decode (`[REQUEST-DECODE-OUTPUT]`): 10.31 tok/s, tokens identical to the earlier runs
  (baseline 10.07).
- MiMo dual-GPU, four prompts: decode 13.65 / 15.00 / 15.38 / 14.87 tok/s, prefill 24.9 / 41.4 / 74.7 /
  77.8 tok/s, tokens identical. So the missing `ZEROMOE_SCHED_EVENTS` (patch 0008) does not change MiMo
  prefill on this upstream.
- **The first run after a fresh ggml build is slow**: the sm_120 GPU JIT-compiles the kernels from PTX.
  Ling ran at 4.49 tok/s and MiMo's first prompt prefilled in 44.7 s (0.47 tok/s); the next run was normal.
  Always do a warm-up run before taking a number from a new build.

## Master build (2026-10-23)

`GGML_REPO=/path/to/llama.cpp scripts/build_ggml.sh build_master_ggml` then
`GGML_ROOT=build_master_ggml/ggml-src/ggml GGML_PREBUILT=build_master_ggml/ggml-build scripts/build_with_ggml.sh build_master`
in the main checkout, from the merged series: `build_master/zeromoe` (RUNPATH inside the main checkout).
Ling short decode on it: 10.39 tok/s warm (4.49 on the first, JIT, run), tokens identical.

Worktree builds (`build_mimo`, `build_ggml` in the feature worktree) are disposable once the branch is merged:
the merged series has no ggml changes beyond master's, so the master build above is the one to keep.

## Open

- `cmake/ZeroMoEGgmlPatches.cmake` is used only when CMake builds ggml itself; the scripts are the
  tested path.
- 0008's `ZEROMOE_SCHED_EVENTS` has no equivalent on this upstream; MiMo's prefill is unchanged without it.
