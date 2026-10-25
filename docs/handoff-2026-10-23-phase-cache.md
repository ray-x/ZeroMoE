# Handoff — shared phase cache, paused 2026-10-23 afternoon

## Stop state and authorization

The user requested: **when the next job/test finishes, pause the session and write an afternoon summary and
handoff.** The latest Ling test has finished. Do not start new builds, model runs, tuning or refactor work
until the user resumes. No Python validation, ZeroMoE, or CMake processes remained when checked.

Working tree: `/home/ray/Projects/ZeroMoE/.worktrees/mimo-v26-flash`

Branch: `feat/mimo-v26-flash`

HEAD: `6ad3a87d80f9d37f36e336e303516e9da428d08b`

No new commits/merges. All afternoon work is uncommitted, mixed with preserved user WIP. Do not reset or
overwrite the tree. The transcript to read is `mimo_tunning.md`. `.ccls-cache/` is unrelated and pre-existing.

The [afternoon summary](afternoon-change-summary-2026-10-23.md) separates inherited changes from this agent's
work. The implementation design is [phase-cache.md](design/phase-cache.md).

## Requested milestone versus actual state

The user authorized completing:

1. Shared, separately releasable prefill/decode phase-cache allocation.
2. Placement-aware graph reuse and preservation of KV/recurrent state.
3. Correctness/memory/transition-cost validation on MiMo and Ling.

**1–2 are implemented, but the milestone is not complete: Ling fails the numerical part of step 3.**
MiMo passes. Keep `ZEROMOE_PHASE_CACHE_MIB` defaulting to zero; do not enable this in baseline profiles,
merge it as fully validated, or weaken the numerical gate to make Ling pass.

## Implementation map

| Component | Files |
|---|---|
| Shared fixed/fit VRAM budgets and embedding placement | `memory/resident_memory_policy.hpp`; `core/engine_config.hpp`, `core/engine_runner_dense.cpp`, `core/gpu_topology.hpp` |
| Actual ready-identity coverage | `models/graph/graph_health.hpp`; MiMo configure path; `models/glm/ggml_glm5next_graph_offload.cpp` |
| DeepSeek native-pipeline adapter | `models/deepseek/ggml_deepseek_v41_graph_decode.cpp`; `models/graph/cold_moe_strategy_cache_backed.hpp`; conformance script |
| Model metadata advisor | `tools/gguf_plan.cpp`, `tools/gguf_tool.cpp` |
| Decode extension and its GPU weight/map buffers | `models/graph/phase_expert_cache.hpp` |
| Route ownership invariant | `memory/phase_route_partition.hpp`; `models/graph/moe_ingraph.hpp` |
| Placement generation/borrower invariant | `memory/placement_epoch.hpp`; `models/graph/graph_core.hpp` |
| MiMo lifecycle and private embedding fix | `models/mimo/ggml_mimo2_graph.cpp` |
| GLM/Ling lifecycle and generation guards | `models/glm/ggml_glm5next_graph.cpp`, `_decode.cpp`, `_run.cpp`, `_build.cpp`, `_impl.hpp` |
| Common opt-in fields | `models/graph/graph_settings.hpp`; default snapshot |
| Per-request diagnostic teacher forcing | `core/generation_loop.hpp` |
| CUDA gate and small regression tests | `bench/phase_cache_transition.py`; `tests/test_phase_routes.cpp`, `test_phase_cache_validation.py` |

Extension storage is separate from the permanent topology cache. Only experts already **ready and pinned
in the offload DDR cache**, and absent from permanent GPU hot sets, are selected. The same identities survive
release/refill cycles. Refill holds a store reader lease and copies three planes without calling `ensure()`
or a model reader. Row strides use `engine.expert_cache_layer_stride_alignment()`; each layer's slot includes
gate/up/down data, with aliases built by the existing `make_alias_weight()` helper.

Graphs lease the active `PlacementEpoch`; retirement refuses live borrowers. MiMo destroys old decode graphs
before release, destroys prefill workspace before extension allocation, and rebuilds on epoch changes.
GLM/Ling preserves expert-source/state handoff, retires the old graph, and similarly binds/checks generations.
KV/recurrent state never belongs to the extension. Decode epochs are 1, 3, 5; releases are 2, 4, 6, with the
last release at shutdown.

The extra branch currently uses the common `extra_branches` hook. Its ordering relative to the early-ready
CPU join is not separately optimized or performance-gated. Do not conflate a memory/correctness milestone
with a confirmed decode optimization.

## Critical failure and what it narrows down

Latest isolated Ling run disables both GPU assist rings in control and candidate:

`logs/phase-cache-ling-isolated-validation/`

| Comparison | Frames | Minimum cosine | Maximum absolute delta |
|---|---:|---:|---:|
| Fixed-cache control repeated short request | 8 | 1.0 | 0.0 |
| Phase-cache repeated short request | 8 | 1.0 | 0.0 |
| Phase versus control, first short | 8 | 0.9932799215 | 2.1136775 |
| Phase versus control, long | 8 | 0.9732831693 | 3.7901227 |
| Phase versus control, repeated short | 8 | 0.9932799215 | 2.1136775 |

The required comparison floor is 0.999. Prefill/step-zero logits match, then decode differs. Repeated
requests are deterministic within each placement, including after the long prefill and reallocation.
This argues against a nondeterministic lifetime/reset problem, but **does not prove the static weight layout,
routing, combination, or quantized GPU computation is correct**.

Adaptive GPU assist made the original control itself vary (minimum repeated-request cosine about 0.998917),
so assist was disabled to isolate the cache. The mismatch remains; assist is not its sole cause.

Next investigation, only after resume:

1. Verify phase-buffer plane bytes against the pinned DDR sources and inspect selected layer/expert/slot
   geometry. No optional byte-verification mode is implemented yet.
2. Compare an actual selected expert's phase alias against a canonical contiguous GGML-CUDA tensor using
   exactly the same input, IDs, weights and quant types. Probe first/middle/last slots. This can separate
   an interleaved-stride/view bug from backend CPU/GPU arithmetic differences. This probe is proposed,
   **not implemented**.
3. Trace phase contributions at the first diverging layer against the old cold contribution. Check masks,
   remaps, top-k slots and application of router weights once. CPU unit tests enforce ownership, but they
   do not replace numerical checks of the CUDA view/kernel.
4. If alias versus canonical GPU matches, quantify CPU versus CUDA behavior for those quant types against an
   independent reference. Do not label the current discrepancy "just rounding" or lower the gate without evidence.
5. Rerun the isolated Ling gate after a fix, then rerun affected MiMo checks if shared math/storage changed.
   Measure steady-state performance only after correctness and the user's baseline restriction are resolved.
6. Review the failure path in `~PhaseExpertCache()`: it currently ignores a false return from `release()`
   before freeing the descriptor. Normal tested shutdown retires all borrowers first, but erroneous owner
   destruction with live borrowers or a CUDA cleanup failure needs a fail-closed lifetime policy.

Teacher-forced generated token agreement is **not** independent accuracy evidence: those tokens are forced.
Compare the saved raw logits at every step. A report checking only step zero is insufficient.

## Passing MiMo result

`logs/phase-cache-mimo-decode-validation/result.json`

- Requests: short → 693-token long prompt (512+181 chunks) → repeated short.
- Compared frames: 8 / 2 / 8, teacher-forced under the saved control streams.
- Minimum cosine: 0.9995852342 / 0.9993794108 / 0.9995852342.
- Maximum absolute delta: 0.9377795 / 0.6457138 / 0.9377795.
- Repeated candidate short request: all eight logits frames bitwise identical.
- Extension: 117 experts; 1,070,393,600 bytes, same across three restores.
- Refill times: 96.309, 89.955, 89.855 ms.
- Release times: 0.627, 0.631, 2.623 ms.
- Decode store loads: 0. Sampled peak GPU memory: 23,628 / 15,356 MiB.

The first attempt exposed an unused MiMo-private embedding upload. Removing it recovered roughly 0.66 GB
and fixed the long-prefill OOM. The common engine's CPU embedding policy alone had not prevented this private copy.

## Ling memory/I/O observations

Latest isolated candidate: 314 experts; 1,071,420,320 bytes; refill 199.129 / 187.744 / 189.369 ms;
release 2.244 / 2.196 / 2.153 ms; sampled peak VRAM 22,138 / 13,958 MiB. Three allocation/release cycles
completed and the graph generation advanced correctly.

Its standard 64 GiB DDR reference profile does not cover the whole model: ready-union coverage was
23,634 / 34,816 expert identities (about 68%). Existing demand I/O is therefore expected for these prompts.
The isolated candidate's final store counters reported 1,159 decode loads. This is **not** a claim of zero
SSD reads for Ling. The extension itself issues no demand loads; its sources are pinned DDR.

The existing 20 GiB host-availability guard occasionally blocked/re-enabled allocations during Ling's
long prefill. The runs remained under their kernel-enforced memory caps. Do not remove these guards.

## Artifact index

All paths are relative to the MiMo worktree. Logs/build outputs are generally ignored by Git; preserve them.

| Artifact | Meaning |
|---|---|
| `logs/phase-cache-mimo-validation/` | Initial control OOM before fixing private embedding upload |
| `logs/phase-cache-mimo-validation-v2/control/` | Valid saved short/long control logits after the fix; old EOS-completion harness stalled, then was stopped |
| `logs/phase-cache-mimo-decode-validation/result.json` | Passing complete MiMo result |
| `logs/phase-cache-ling-control/control/` | Original assist-enabled control |
| `logs/phase-cache-ling-validation/result.json` | **Superseded prefill-only report; not proof of decode correctness** |
| `logs/phase-cache-ling-decode-validation/` | Assist-enabled all-decode run; failed repeated-request numerical check |
| `logs/phase-cache-ling-isolated-validation/` | Latest no-assist control + candidate; final numerical failure |
| `/tmp/zeromoe-phase-ling-isolated-validation.log` | Traceback from the latest strict gate (fails at first short request; long mismatch also measured separately) |
| `/tmp/zeromoe-phase-mimo-decode-validation.log` | Passing MiMo transcript |
| `/tmp/zeromoe-phase-teacher-build.log` | Last successful engine build after adding per-request teacher forcing |
| `/tmp/zeromoe-phase-reference` | Older separate reference binary used for the assist-enabled Ling control; not needed for latest isolated controls |

Each completed case has `request-0.bin`, `request-1.bin`, `request-2.bin`, token files, `engine.log`,
`console.log`, and `vram-samples.json`. MiMo's reused control has only requests 0 and 1; the repeated short
request's expected logits/tokens intentionally come from request 0. Do not invent a measured third control.

## Build and tests for resumption

The candidate is `build_mimo/zeromoe`, last built around 16:27 AEDT. It links the patched ggml in this
same worktree:

```text
ZEROMOE_GGML_ROOT=<worktree>/build_ggml/ggml-src/ggml
ZEROMOE_GGML_PREBUILT_DIR=<worktree>/build_ggml/ggml-build
```

Do not use another worktree's disposable prebuilt libraries, delete these build directories, or rebuild a
binary while a test is executing that same file. CUDA access/NVML fails inside the sandbox but works outside;
GPU validation required approved execution outside the sandbox. Do not reset drivers to work around this.

After resume, CPU/build checks are:

```bash
cmake -S . -B build_mimo
cmake --build build_mimo -j4 --target zeromoe graph_settings_test layer_plan_test phase_route_test resident_memory_policy_test
build_mimo/graph_settings_test tests/graph_settings_defaults.snapshot
build_mimo/layer_plan_test
build_mimo/phase_route_test
build_mimo/resident_memory_policy_test
python3 tests/test_gguf_embedding_plan.py
python3 tests/test_phase_cache_validation.py
bash scripts/check_moe_conformance.sh
git diff --check
```

These checks passed when last run. The layer-plan test intentionally logs an ERROR for a rejected non-I32
survey input; its final `layer_plan_test: OK` is the success criterion. CMake target additions occasionally
needed an explicit configure before the new target was visible.

After making a justified Ling fix, reuse its isolated control with a fresh output directory:

```bash
python3 bench/phase_cache_transition.py --model ling --isolate-assist \
  --control-output logs/phase-cache-ling-isolated-validation/control \
  --output logs/phase-cache-ling-next-debug
```

For a shared fix affecting MiMo:

```bash
python3 bench/phase_cache_transition.py --model mimo \
  --control-output logs/phase-cache-mimo-validation-v2/control \
  --output logs/phase-cache-mimo-next-debug
```

Do not run these commands during the requested pause. The harness refuses an existing output directory,
uses real memory caps, compares all frames, and fails rather than writing `result.json` on a numerical failure.

## Constraints and remaining broader work

- Do not rerun the frozen model baseline suite yet; the user explicitly deferred it in the transcript.
- Do not use SSD offload when testing GLM 5.3 Flash. This afternoon's CUDA cases were MiMo and Ling, not GLM Flash.
- No confirmed steady-state tok/s improvement or regression. Eight-token correctness runs and transition
  timings are not a performance rebaseline.
- Keep phase-cache default zero. Only MiMo and GLM/Ling currently integrate this allocator; Qwen and DeepSeek
  are not migrated to phase reuse merely because they share `GraphCore`.
- The native cache-backed adapter is live, but reusable asynchronous lifecycle ownership, named staging/pilot
  policies, automatic hardware calibration/PCIe-cost selection, and complete family migrations remain unfinished.
- The proposed canonical-GPU/byte probes were not implemented before pause; do not assume a verification
  flag or probe target exists.

## Update: Ling mismatch attributed to placement noise (evening)

Teacher-forced against llama.cpp (CPU, same 8 forced tokens, short request, `tools/ling_llama_logits.cpp` with
`LLAMA_FORCE_TOKENS`), cosine per decode frame:

| Run | Frames 0-7 | Top-1 |
|---|---|---|
| fixed-cache control | .9703 .9764 .9819 .9905 .9958 .9940 .9940 .9969 | 8/8 |
| equivalent permanent placement (3090 cache 4.07 GB) | .9628 .9746 .9803 .9904 .9958 .9956 .9973 .9955 | 8/8 |
| phase-cache extension | .9703 .9773 .9836 .9926 .9959 .9953 .9971 .9964 | 8/8 |

The extension is as close to llama.cpp as the control; no phase-specific error. Ling IQ1_S cannot reach 0.999 against any
other placement or against llama.cpp (CPU vs CUDA kernels differ ~1% per op); the 0.999 floor applies to MiMo.
Pairwise drift between any two Ling placements is 0.96-0.99. Decode cost: ~8-12% below an equivalent permanent
placement with identical SSD loads (extra per-layer branch). The gate run's 10.32 vs 8.04 tok/s is confounded by
diverging tokens (loads 133 vs 441). Extension stays opt-in (default 0). Not done: first-diverging-layer trace,
kernel profile, merging the extension into the hot branch.

## Correction (later the same evening): kernels are exact; the drift is activation rounding

The "placement-noise band" idea above is withdrawn. Findings:

- Stock llama.cpp (unpatched, CUDA build `build-cuda-stock`), CPU vs CPU+10 GPU layers, same forced tokens:
  cosine .952 .969 .978 .987 .994 .994 .994 .996. The Ling drift is not ZeroMoE-specific.
- Node-by-node diff (`LLAMA_NODE_DUMP`, `tools/node_dump_diff.cpp`): identical until the first GPU layer; dense MUL_MAT
  0.2-0.4%; routed-expert MUL_MAT_ID 0.9% (gate/up), 1.4% (swiglu), 2.3% (down).
- `scripts/check_kernel_numerics.sh` (`EXACT_INPUTS=1`): with activations 8-bit rounding represents exactly, CPU and CUDA
  match float64 to <=4e-5 for IQ1_S, IQ2_XXS, IQ2_S, Q2_K, Q3_K. The kernels are correct. With ordinary inputs both
  deviate 0.5-0.9% from exact math: CPU rounds activations in 256-value Q8_K blocks, CUDA in 32-value Q8_1 blocks.
  Expert chains of 1-bit weights amplify it; MiMo's Q2_K/Q3_K amplify it far less (0.9994+).
- Consequence: no gate can hold Ling at 0.999 across CPU/GPU placement changes unless both backends round activations
  identically (an upstream ggml change, untried). Gate proposals: the kernel check above, plus same-placement
  determinism (bitwise repeats). The 0.999 floor is unchanged for MiMo. Extension still opt-in, 8-12% slower on Ling.

## Ling decode cost, clean measurement (32 tokens, true teacher forcing, assist off)

`bench/phase_cache_transition.py --model ling --isolate-assist --ngen 32`: control 3.90/3.82/5.40 tok/s, loads 3402, misses 4707;
extension 3.80/3.72/5.26 tok/s, loads 3424, misses 4726 (about -2.5-3%, same SSD traffic). nsys: whole-run GPU kernel time 86 vs 87 ms.
The earlier gate-run slowdown (10.3 -> 8.1 tok/s) is a route-divergence artifact: the runs were free-running (they set ZEROMOE_TEACHER_FORCE_TOKEN_FILE, which bench/ling/reference-audit/run.sh unsets; the wrapper takes the token file in LING_TOKEN_FILE, so forcing through it was not actually tested),
the extension perturbs numerics, 11/68 layers pick different experts at the first decode step, and the text then diverges
(more cold experts, +300 demand loads). Cosine vs control over all 32 frames: min 0.9799 (Ling's known CPU/GPU placement drift).

## Decision (user, 2026-10-23): Ling's gate is the kernel check

Ling no longer needs cosine >= 0.999 against a differently-placed control. Its phase-cache gate is
`scripts/check_kernel_numerics.sh` (aliased == contiguous; CPU/CUDA within 1e-4 of float64 on exactly representable inputs)
plus a bitwise-identical repeated short request after the prefill/decode transitions (enforced in
`bench/phase_cache_transition.py`; the ngen=32 run satisfies it: cosine 1.0, max_abs 0). MiMo keeps the 0.999 floor.
Measured cosine against the control is still recorded in result.json (Ling: min 0.9799 over 32 frames).

## Real-weight check (`tests/test_real_tensor_numerics.cpp`, layer 63, real Ling experts, approximately real activations)

Rel L2 against float64 on the same quantized weights (CPU / CUDA / CPU-vs-CUDA): gate IQ1_S 0.93% / 0.81% / 1.16%;
up 0.99% / 0.87% / 1.23%; swiglu 1.31% / 1.23% / 1.70%; down IQ2_XXS fed the SAME exact input 1.71% / 0.94% / 1.76%;
down on each backend's own chain 2.16% / 1.55% / 2.51%. These reproduce the llama.cpp node diff (0.93/1.4/2.3%).
The CPU down-projection is about twice as far from exact as CUDA's on real swiglu activations (Gaussian test: both ~0.7%).
Likely reason (inference, not isolated): swiglu activations are spiky, and CPU's 256-value Q8_K scale is set by the outlier,
so the other values round coarsely; CUDA's 32-value Q8_1 blocks adapt. Caveat: the test inputs are not an exact replay of
the graph (the dump's own gate output is not reproduced), so absolute numbers are approximate.

### Why the CPU down-projection is less accurate (proven)

Exact float64 math on the swiglu output after rounding it to 256-value blocks (CPU's Q8_K scheme) gives rel error 1.71e-2,
identical to the CPU kernel's measured error; rounding to 32-value blocks (CUDA's Q8_1 scheme) gives 9.44e-3, identical to
CUDA's. So the whole difference is activation rounding. The swiglu output is spiky (block max / block rms = 7.3, Gaussian ~3),
so the one outlier-driven scale per 256 values rounds the other values coarsely. The swiglu op itself adds no error; its
1.3% is inherited from gate/up.

## Prototype: CPU rounds activations in 32-value blocks (resolves the Ling CPU/GPU drift)

`patches/experimental/ggml-cpu-act32-prototype.diff` (against llama.cpp 8baabd53b; built in a separate git worktree
`/home/ray/Projects/llama.cpp-act32`, CUDA unchanged, not part of the ZeroMoE ggml series). For IQ1_S, IQ2_XXS and
IQ2_S the CPU dot product uses a Q8_0 activation (32-value blocks, like CUDA's Q8_1) and exact dequantized weights; the
"tiled" CPU kernel is excluded for those types (it assumes a Q8_K activation).

Ling, 8 teacher-forced frames, cosine between a CPU-only run and a run with 10 of 73 layers on the 3090:

| Build | Frames 0-7 |
|---|---|
| stock llama.cpp | .9522 .9689 .9775 .9865 .9943 .9939 .9944 .9956 |
| patched CPU (32-value blocks) | .9999 .9998 .9999 1.0 1.0 1.0 1.0 1.0 |

So the placement drift was CPU activation rounding. Caveats: the prototype dequantizes each weight row per dot (slow; fine for
a numerics experiment, not for production); only the three Ling expert types are patched; stock CPU vs patched CPU is
0.96-0.99, i.e. the stock CPU path is the outlier against CUDA's finer rounding. A production version would need a real
32-block vec_dot for these types (and the same for other IQ/K-quant types) and a decision on changing CPU numerics.
The prototype's first run produced all-zero logits because the tiled kernel misread the Q8_0 activation; that is fixed by
excluding the three types from `ggml_tiled_supported`.

### Option 1 result: only the down-projection input needs fine blocks

`patches/experimental/ggml-cpu-act32-down-only.diff`: 32-value activation blocks for IQ2_XXS only (Ling's ffn_down_exps);
IQ1_S (gate/up) and IQ2_S stay on the stock Q8_K and tiled paths. CPU vs CPU+10 GPU layers, 8 teacher-forced frames:
.9999 .9999 .9999 .9999 1.0 .9999 .9999 1.0 (stock .9522 ... .9956; all-three-types .9999 ... 1.0). The whole drift is the
down-projection input. Still a slow prototype (dequantizes rows); speed has not been measured on a real kernel.

### Decision (user): the finer-activation CPU kernel is not acceptable for production

Micro-benchmark of a vectorized IQ2_XXS x Q8_0 kernel (decode shape, K=1280 N=4096, 8 of 64 experts, 1 token): stock 739 us
vs 800 us at 10 threads (+8%); 3.80 ms vs 4.06 ms at 1 thread (+6-7%). Cause: one int->float convert + FMA per 32-value
sub-block instead of per 256, and the tiled kernel disabled for the type. A 7-8% slowdown on the CPU expert path is not
acceptable for ZeroMoE, so the change stays an experiment (`patches/experimental/*.diff`), never in the production ggml series.
Possible validation-only use: an offline build for placement-invariant numerics checks. Ling decode with the kernel was not timed.
See also docs/experiments/ggml-cpu-activation-rounding.md for the CPU activation-rounding research (patches in patches/experimental/, not applied).
