# Full-graph CUDA MoE onboarding

This is the authoritative workflow for adding or changing a CUDA MoE family. It is designed to catch architecture, numerical, and performance mistakes before they turn into hours of downstream debugging. Follow the stages in order and keep the model in progress until every applicable gate passes.

## Stop/go onboarding sequence

Use the fillable [model profile transfer record](model-porting-profile-transfer-template.md) for every new model. Complete its setting matrix and frozen-reference block before the first long run; save the completed copy beside the run artifacts.

### 1. Establish the GGUF and execution path before coding

Record the exact GGUF path, architecture identifier, quantization, layer count, expert count, top-k, shared-expert topology, attention/state types, and required tensor shapes. Do not infer compatibility from the model name or a shared `ModelFamily` label.

Trace the actual code path:

1. `GGUFFile::load_header` reads model metadata and tensor locations.
2. `detect_model_family()` identifies the family.
3. `resolve_family_ops()` resolves its graph and model operations.
4. `full_graph_model()` selects the registered `FullGraphModel` adapter.
5. The adapter validates the exact `general.architecture` and tensor topology before large allocations.
6. `plan_decode_memory()` reports the graph memory plan before placement and execution.

Before coding, save these results in the per-model onboarding note. If the exact architecture has no registered graph adapter, implement that adapter first. The runtime must reject the architecture until then. A family label must never silently authorize an architecture the adapter has not validated.

#### Inherit the nearest measured performance profile

Before selecting defaults or starting a smoke run, identify the closest accepted profile by physical GPU, architecture, quantization, expert footprint, active parameters, and offload strategy. Record its config, command, measured prefill/decode rates, CPU thread count, placement, usage-survey source, hotset bytes, pinning, and cache/promotion settings. Map each setting to the new adapter and explain any setting that cannot carry over. Engine defaults do not prove that accepted tuning has transferred.

Use the profile's target hardware for performance runs. A run on another GPU can establish topology or correctness, but it does not establish target-profile performance. If the target GPU or host is busy, wait for an uncontended slot rather than substituting another card or lowering CPU threads; record exploratory results separately.

Portable candidate configs should use `[[gpus]].device_id = -1` to select the visible device with the highest VRAM/performance score. Use an explicit ID only for a deliberate hardware-specific profile. Leave `omp_max_threads = 0` for host-relative sizing and record the resolved GPU and effective CPU thread counts from startup logs. The process-wide OpenMP default uses `hardware_concurrency() - 2`; the full-graph CPU backend may choose physical cores from the process affinity mask. Do not copy a thread count from another host.

The binary's CUDA targets must include every GPU that automatic selection may choose. Record the configured target list and inspect the built CUDA library (for example, with `cuobjdump --list-elf`). If a target is absent, rebuild for it or restrict the selectable device set before loading the model. Do not discover a GPU-target mismatch after the model and graph have been allocated.

Expert counts and total GGUF bytes do not predict the active MoE work. Record per-layer expert count, router top-k, gate/up/down shapes, quantization, bytes per expert record, and usage coverage. Estimate active routed parameters and bytes per token from the actual top-k and projection tensors across layers; two similar-size models can route substantially different work. Carry over a VRAM byte budget only after recalculating per-expert bytes, and generate a model-specific routing survey rather than reusing another model's expert rankings.

For Qwen 3.8 on RTX 3090, measured references include a 22-GPU/26-CPU layer split at 10 CPU threads (28.66 tok/s) and a usage-ranked global hotset of 14,554 Q2_0 experts at 10 threads (43.12 tok/s). These are distinct strategies and quantizations; transfer the setup, then measure the new model's equivalent placement and coverage. See [`qwen38-flash.md`](../models/qwen38-flash.md) and [`perf-baselines.md`](../../configs/bench/perf-baselines.md).

#### Verify every quantized expert execution path

For every target MoE layer, enumerate the exact `(gate, up, down)` GGML types
from the GGUF. A summary such as “GPU-capable” does not establish that the
hot split, cold CPU path, or a fused CPU kernel supports that tuple.

For each distinct type tuple, record the effective runtime path: CUDA resident
expert, generic CPU `MUL_MAT_ID`, fused CPU cold-expert node, staged upload, or
another explicit implementation. Include the GGML CPU `vec_dot` type and
activation quant type where a fused path depends on them. A rejected fused
tuple must be logged as a fallback and timed as its own profile; do not treat
family-level sharing as inherited tuning.

Add a bit-exact fused-versus-stock check for each tuple the adapter can select,
including real dimensions, top-k, route masks, and the single-token decode
shape. In the integration log, report how many layers selected each kernel
path. If the selected path differs from the reference profile, block
performance acceptance until that effective path has its own matched timing.

#### Transfer placement maps, not just layer counts

When an accepted profile uses `cpu_affinity_layers`, carry that exact layer map together with its candidate `graph_device_expert_layers` block and trailing/leading direction. The graph planner applies CPU-affinity exclusions before selecting device-resident layers, so a block value is not the number of GPU layers. Record the planner-selected layer set and arena bytes. If the new model cannot fit the same set, state the memory-driven difference and benchmark that effective placement; do not replace it with an arbitrary contiguous tail.

For every copied config or environment switch, trace the consumer to the actual adapter execution path. Separate full-graph builder/runtime settings from flags consumed only by legacy layer-by-layer dispatch. A legacy `[CONFIG] ... off` line does not mean full-graph capability is off; verify `[GRAPH-CAPS]`, the planner result, and graph-gate replay counters.

Run `tools/compare_full_graph_profiles.py` against the measured reference
and candidate before model load. Pass reference run CLI/harness values with
`--reference-override`, and pass candidate effective settings from startup
logs or hardware policy with `--candidate-override`. Use
`--strict-placement` to catch a changed eligible-layer map before memory
planning. The script reviews profile transfer; it does not predict VRAM fit.

### 2. Map reference operations before building the adapter

Use llama.cpp as the initial executable specification. Record its source path and commit, then map each model block stage to a ZeroMoE tensor/op boundary. The map must include:

- Input/output tensor names, shapes, layouts, and dtypes.
- Normalization function, epsilon, reduction dimensions, and scale placement.
- Projection and activation order, including gate/up/SwiGLU operations.
- Residual order, router normalization, expert selection/weights, and state handoff.
- The corresponding source function in llama.cpp and adapter function in ZeroMoE.

Preserve llama.cpp's operation sequence while establishing parity. Do not replace an operation with an algebraically equivalent expression, or fuse/reorder operations, without a same-input boundary-trace A/B. Floating-point rounding differences can accumulate through quantized FFNs and change near-tie expert routes.

### 3. Implement the registered full-graph adapter

Keep model-specific equations, tensor validation, and state in the adapter. Reuse `GraphCore`, shared routed-MoE builders, placement, cache, and offload strategies. Do not add a CUDA MoE family through the legacy layer-by-layer expert executor.

The adapter must declare its graph capabilities, exact supported architectures, decode memory plan, device expert placement, CPU-lane support, state ownership, and applicable runtime tuning. `Unsupported` means the GGUF architecture or tensor topology failed validation; startup must fail before large graph allocation. It never authorizes a memory fallback.

### 4. Prove placement and graph execution before benchmarking

The decode memory plan is mandatory evidence:

- `Fits` requires positive required and available device-byte estimates with required <= available. Graph decode is mandatory.
- `Unknown` does not authorize fallback. Graph decode remains mandatory.
- `Insufficient` permits layered fallback only when required > available, the reason is logged, and the adapter supports that fallback. Graph-only models cannot fall back.

When `graph_preflight_strict=true`, an `Insufficient` memory plan is a startup
error and cannot enter layered fallback. Use this setting for a full-graph
onboarding or performance run so an oversized hotset fails before prompt
generation rather than producing a misleading layered result.

The plan must include dense and expert weights, KV/recurrent state, and graph workspace—not only the GGUF file size or an MoE cache estimate. If the planner says the full graph fits, graph build or execution failure is fatal. Never silently run the legacy executor.

For a full-fit run, require:

```sh
python3 tools/check_full_graph_moe_run.py <run.log> --full-fit
```

For Qwen3.5/Qwen3.6, use the gate matching the declared placement scope. `--require-all-expert-layers-on-device` applies only when the profile claims every routed expert layer is resident in VRAM. It is not the default acceptance condition for this offloading engine: a valid VRAM/DDR profile must report the expected device block or usage hotset, a complete admitted DDR source, `memory_plan=fits`, and zero layered/per-expert fallback.

`--full-fit` means the graph's required device memory fits its available device memory. It does not mean the entire model or all expert weights fit in VRAM. For a partial VRAM/DDR full-graph run, require graph prefill as well:

```sh
python3 tools/check_full_graph_moe_run.py <run.log> --full-fit \
  --require-full-graph-prefill
```

When `--require-full-graph-prefill` is set, every prefill must remain on the
full graph. A logged prefill graph-build failure followed by legacy prefill
fallback fails the gate, even if another request in the same log completed a
full-graph prefill. The decode memory plan and the batch-specific prefill
workspace check are separate admission checks.

Use `--require-all-expert-layers-on-device` only for a separately declared all-VRAM profile.

This gate checks graph capability, a `Fits` plan, successful graph decode replays, and zero one-token layered forwards or legacy per-expert dispatches. If the model genuinely does not fit, test the explicit memory-fallback path separately with `--allow-memory-fallback` and verify its logged reason.

### 5. Compare teacher-forced numerical boundaries before optimizing

A coherent sample is only a smoke check. Teacher-force the same prompt and decode token sequence through llama.cpp and ZeroMoE so autoregressive divergence cannot hide the first numerical mismatch. Compare each layer's pre-MoE hidden state and routed expert IDs/weights. For recurrent models, include state inputs/outputs and the prefill-to-decode handoff.

#### If the comparison differs: debug in this order

1. **Freeze the experiment.** Record the exact GGUF and hash, reference and ZeroMoE commits, build/patch provenance, device, context, positions, prompt token IDs, forced decode tokens, thread count, optional features, and graph policy. Confirm both runs use the same prompt and that a full-fit ZeroMoE run passed the graph gate. Use an otherwise idle GPU and fresh trace directories so stale or appended dumps cannot be mistaken for the current run.
2. **Replay identical inputs.** Derive the forced token sequence from one reference run and feed it to both engines. Do not compare free-running text after the engines have chosen different tokens. Capture the same token positions and tensor shapes in both runs, preserving raw values and metadata.
3. **Find the earliest divergent boundary.** Compare in execution order: token IDs/embeddings; layer input and pre-attention norm; attention or GDN projections, normalization, RoPE/convolution and state; attention output and residual; post-attention norm/MoE input; router logits and probabilities; top-k IDs and normalized weights; routed gate/up/activation/down; shared expert; combine and residual; next-layer hidden state. First locate the earliest layer whose boundary differs, then add detailed taps only inside that block.
   The comparator is part of the gate: it must return nonzero on the first mismatch and save that boundary's tensors, metadata, prompt IDs, and forced token IDs. A report that lists mismatches but exits zero is not a pass. Validate the comparator against a known differing trace before trusting it.
4. **Diagnose from the first mismatch.** If layer input differs, inspect the preceding layer's attention/GDN output, residual, recurrent state, and position/state handoff. If the MoE input matches but router logits do not, inspect the router projection weights, matrix orientation/layout, dtype, accumulation, and normalization. If logits match but IDs differ, inspect top-k tie-breaking, expert indexing, and order; record the margin between the last selected and first excluded experts. If IDs and weights match but expert output differs, align slots by expert ID and compare gate, up, activation, and down separately, including quantized kernel/accumulation behavior. If expert outputs match but the next hidden state differs, inspect shared-expert weighting, routed-weight application, combine order, residual order, and state/cache writes.
5. **Quantify and fix one cause at a time.** Compare tensors only after confirming shape, layout, dtype, and token row match. Report relative RMS, maximum absolute error, and cosine similarity; for routing, also report exact expert IDs/weights and top-k boundary margin. Treat a route-ID mismatch or an unexplained error as unresolved. Check the llama.cpp source operation order before changing code. Make the smallest correction at the first divergent operation, then rerun the same trace before broadening it to more layers or tokens.
6. **Re-run acceptance after parity is restored.** Repeat the teacher-forced comparison, full-fit graph gate, and matched performance gate on the final binary. Do not tune later layers or accept a speed result collected from a fallback path while correctness is unresolved.

For usage-ranked offload, collect the reference token sequence first and
use the same teacher-forced prompt/token sequence in both engines. Save
llama.cpp's per-layer routed IDs and selected weights as the immutable
reference. Compare every ZeroMoE configuration against that reference
before using its routed-expert histogram to rank a VRAM hotset. A long
survey can also be the parity workload: derive route usage only from
tokens whose per-layer IDs and weights pass comparison. A short trace is
a fail-fast reproducer, not a replacement for parity across the survey
workload.

For Qwen, `ZEROMOE_QWEN4EXP_PROBE_DUMP=<directory>` writes graph taps, including layer inputs, router logits/probabilities, and GDN probes. By default it retains taps from all layers; set `ZEROMOE_QWEN4EXP_PROBE_DUMP_LAYER=<layer>` to retain roots and detailed expert stages from one layer while locating the first mismatch. Without that selector, `ZEROMOE_QWEN4EXP_PROBE_LAYER=<layer>` selects the detailed routed gate/up/activation/down taps. These diagnostic roots may change graph scheduling, so use them only for correctness isolation and compare against a minimally instrumented rerun before treating a difference as production behavior. Use `ZEROMOE_TEACHER_FORCE_TOKEN_FILE=<file>` with `ZEROMOE_TEACHER_FORCE_SKIP_SAMPLING=1` to replay the same decode tokens. Use the matching llama.cpp evaluation-callback or reference trace to capture the same boundaries. Other model families should expose equivalent adapter-specific taps rather than relying on output text alone.

The common `GraphCore` diagnostic is enabled with
`ZEROMOE_FULL_GRAPH_PROBE_DIR=<directory>` and optionally narrowed with
`ZEROMOE_FULL_GRAPH_PROBE_LAYER=<layer>`. Adapter builders register existing
layer inputs/outputs and dense/router tensors with
`capture_full_graph_boundary`; each graph run calls the shared dumper after
scheduler synchronization. This gives new adapters the same raw tensor and
metadata format across model families. Add architecture-specific recurrent
state taps at the adapter boundary when the state handoff is relevant. Keep
this trace diagnostic-only and compare it with a minimally tapped replay.

### 6. Benchmark only the validated graph path


Before the first performance run, print a startup summary of the expected profile versus the adapter's effective settings: physical device, requested and planner-selected device layers, CPU/DDR expert bytes, usage-hotset count/bytes/traffic coverage, process and graph-lane thread counts, pin budget, prefill chunk, and enabled promotion/offload paths. A defaulted or missing setting is a failed profile-transfer check until explicitly accepted. Do not accept a partial-layer smoke rate as a performance test for a 22/26 profile or usage-ranked hotset profile.

#### Interpret usage hotsets against layer timing

The shared Qwen expert-byte planner spends a fixed budget on global route
counts. This maximizes estimated total route coverage, but does not necessarily
minimize the slowest serial CPU lane. An Ornith 80% per-layer coverage-floor
prototype raised minimum layer coverage from 73.48% to 80.03% at nearly the
same total coverage, but two runs measured 33.70 and 33.53 tok/s versus
34.44 tok/s for the global batch-4 control. CPU pressure differed and the
slowest lane shifted to layers 31, 35, and 39. The runtime knob was removed;
its offline simulation remains an experiment record, not an accepted planner
policy.

For a future placement change, preserve per-token routed expert bundles in the
survey so candidate placement can model each layer's cold-expert count, then
compare the predicted critical lanes with uninstrumented CPU/GPU lane timing.
Do not add a layer-balancing rule based only on marginal expert counts; keep a
new objective only after repeated full-graph performance gains and teacher-
forced routed-output parity.

#### Diagnose full-graph timing

Legacy per-request attention, MoE, and disk counters may report zero on a full-GGML-graph request because that path does not pass through the legacy dispatch instrumentation. Use `[PREFILL-DONE]`, `[REQUEST-DECODE-OUTPUT]`, `[REQUEST-DECODE-WALL]`, `[STATS]`, graph build/upload timings, and graph replay counts as the authoritative boundaries; do not interpret zero legacy counters as zero graph work.

The Qwen scheduler evaluation callbacks are invasive. `ZEROMOE_QWEN4EXP_LAYER_TIMING=1` enables CPU node timing; `ZEROMOE_QWEN4EXP_LAYER_TIMING_VERBOSE=1` dumps every CPU/CUDA node and backend (set `log_level = "DEBUG"` to see its detailed records). Either callback can split CUDA graph work into per-node submissions, preventing fused `topk_moe_cuda` and weighted-MoE reduction kernels and changing both latency and rounding. `log_level = "DEBUG"` alone must not enable either callback. Keep both timing flags unset for correctness baselines and all performance acceptance runs; use them only in separate diagnostic runs.
For CPU per-layer timing without the scheduler evaluation callback, `ZEROMOE_QWEN4EXP_COLD_FUSED_TIMING=1` aggregates timing inside each fused cold-expert CPU node. It adds barriers only to that diagnostic node; do not use the timed run for throughput gates.

Benchmark after the full-fit graph gate and teacher-forced parity checks. Use the same GGUF, physical GPU, context, active prompt length, generated token count, and thread count in ZeroMoE and llama.cpp. For ZeroMoE `zeromoe bench --bench-random-tg N`, decode starts from one prompt token; match it with llama-bench `-pg 1,N`. For prefill `--bench-random-pp P`, compare with llama-bench `-p P`.

Without a user-set target, steady-state decode must reach at least 90% of llama.cpp. A user-set absolute floor must also pass via `--minimum-tps`. Keep the model in progress below either applicable threshold; a graph gate or coherent output is not a performance pass.

When the user gives profile-specific absolute targets, keep one run/config per profile and pass its floor with `--minimum-tps` (for example, the documented Qwen 3.8 split and hotset targets are 28 and 42 tok/s). Do not call a result accepted because it meets the relative llama.cpp bar while missing the user's absolute profile target.

Run the relative and absolute performance gate with the matching workload:

```sh
python3 tools/check_full_graph_moe_perf.py <zeromoe.log> <llama-bench.json> \
  --n-prompt <n> --n-gen <n> --context <ctx> --n-threads <n> \
 --n-gpu-layers <n> --n-cpu-moe <n> --device <device> [--minimum-tps <tps>]
```

Report prefill throughput, graph construction/upload, steady-state decode, and end-to-end request rate separately. Keep profiler runs separate from uninstrumented throughput measurements. If llama.cpp cannot provide a comparable reference, state why and establish a quantitative model-specific bar before calling onboarding complete.

### Required timing attribution

Every performance investigation must save a timing report with four views:

- **Per layer and operation:** identify the layer, backend, operation, wall time, and whether the work was dense, attention/GDN, routed expert, or shared expert. For MoE, include selected expert IDs or hot/cold counts where those affect placement.
- **GPU:** report device-side kernel or CUDA graph-node duration, GPU copies, and relevant synchronization. A host callback around CUDA graph submission measures enqueue time, not device execution time. Use Nsight Systems or device-side events for GPU durations; keep the trace as a separate diagnostic run.
- **CPU:** report CPU graph-node time by layer and operation, expert fetch or staging work, and the time between CPU nodes. Label the latter as wait, dispatch, or barrier time only when the trace can distinguish those causes.
- **Request workflow:** report graph build/upload, prefill, decode setup, time to first token, steady-state decode, decode wall time, and end-to-end request time from the same workload record.

Keep inclusive wall time separate from summed CPU/GPU work because those backends can overlap. Do not infer layer or GPU time from `nvidia-smi` utilization samples or legacy dispatch counters. If a required view is unavailable, record the missing instrumentation and keep the performance cause unresolved.

### 7. Preserve reproducible provenance and artifacts

For each accepted measurement, record both project commits, the shared upstream GGML base revision and version, ZeroMoE's applied GGML patch manifest, CUDA toolkit and driver, compiler, and native GPU code targets. Verify a copied GGML build tree's generated `GGML_COMMIT` names the pinned upstream GGML revision rather than the enclosing ZeroMoE commit. Project-only patches are part of the measured implementation; do not call the builds kernel-identical.

Keep binaries, adjacent runtime libraries, benchmark JSON, gate logs, and compact parity evidence in persistent workspace storage. A build under `/tmp` is temporary; copy the complete runtime bundle back and make its library lookup relocatable before relying on it. If upstream base, toolchain, or GPU target differs, mark the ratio provisional and rebuild a matched reference before acceptance.

## Definition of done

Call onboarding complete only for the explicit validated scope: exact GGUF architecture and quantization, device/GPU, context range, and exercised input path. Do not generalize a text-only or single-quant result to untested multimodal paths, other quants, or other devices. Record whether MTP or other optional paths were exercised.

All applicable evidence below must be saved in the model note and persistent run artifacts before changing its status to supported:

- **Architecture:** the exact GGUF architecture resolves to a registered adapter; topology validation passes before large allocation.
- **Memory and execution:** the live planner reports positive required and available bytes. For `Fits`, `check_full_graph_moe_run.py --full-fit` passes with graph decode enabled and zero layered/per-expert dispatches. A graph build or execution failure is fatal. If the plan is `Insufficient`, only the documented memory fallback may be accepted; record that scope as memory-limited and do not claim full-fit support. `Unknown` is never an acceptance result.
- **Correctness:** teacher-forced per-layer pre-MoE hidden states and routed expert IDs/weights match llama.cpp for the same prompt and forced tokens. Stateful models also match the recurrent-state and prefill-to-decode handoff boundaries. Resolve any mismatch before acceptance.
- **Performance:** the full-fit graph gate passes, then repeated matched decode measurements pass both the default 90% llama.cpp ratio and any user-set `--minimum-tps`. Record sample range and compare the same GGUF, device, context, prompt/generation lengths, and thread count.
- **Report and provenance:** report prefill, graph construction/upload, steady decode, and end-to-end rates separately. Record both project revisions, GGML base/version and applied patch manifest, CUDA toolkit/driver, compiler, and native GPU targets. Keep the benchmark logs, JSON, parity summary, and runnable binary/runtime bundle in persistent storage.

A coherent response, successful compilation, one successful graph replay, or passing only one gate is not completion. If any required evidence is missing or fails, leave the model in progress and state the exact missing gate. Mark it supported only after the complete evidence set passes for the declared scope.

## Shared runtime and code boundaries

`FullGraphModel` in `models/graph/full_graph_model.hpp` owns graph enablement, bootstrap, build, run, and shutdown. `FullGraphModelBase` centralizes request validation and prefill/decode dispatch. The registry boundary is `full_graph_model()` in `models/graph/full_graph_models.cpp`.

The startup path reads GGUF metadata, detects the architecture, validates family topology, resolves `FamilyOps`, and selects the registered adapter. The adapter's capabilities and runtime tuning are resolved before cache planning and preload. The graph then binds GGML tensors, schedules supported CPU/GPU work, and connects routed experts to shared cache or offload providers.

The adapter must declare device expert layer count and placement direction, CPU-lane support, host-memory requirements, cache ownership, and prefill limits. Recurrent state must flow consistently across prefill and decode. Runtime tuning must be applied before cache sizing and preload so placement and resource defaults use one resolved configuration. Startup diagnostics should expose adapter capabilities, the resolved memory plan, tuning policy, and final placement.

Shared placement, cache, routed-MoE, and state-handoff behavior belongs in the graph runtime. Keep equations and architecture-specific state beside the model adapter. A successful graph build proves execution only; correctness and performance require their own gates above.
