# Full-graph CUDA MoE onboarding

This is the authoritative workflow for adding or changing a CUDA MoE family. It is designed to catch architecture, numerical, and performance mistakes before they turn into hours of downstream debugging. Follow the stages in order and keep the model in progress until every applicable gate passes.

## Stop/go onboarding sequence

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

The plan must include dense and expert weights, KV/recurrent state, and graph workspace—not only the GGUF file size or an MoE cache estimate. If the planner says the full graph fits, graph build or execution failure is fatal. Never silently run the legacy executor.

For a full-fit run, require:

```sh
python3 tools/check_full_graph_moe_run.py <run.log> --full-fit
```

For Qwen3.5/Qwen3.6, also require all expert layers on device and full-graph prefill:

```sh
python3 tools/check_full_graph_moe_run.py <run.log> --full-fit \
  --require-all-expert-layers-on-device --require-full-graph-prefill
```

This gate checks graph capability, a `Fits` plan, successful graph decode replays, and zero one-token layered forwards or legacy per-expert dispatches. If the model genuinely does not fit, test the explicit memory-fallback path separately with `--allow-memory-fallback` and verify its logged reason.

### 5. Compare teacher-forced numerical boundaries before optimizing

A coherent sample is only a smoke check. Teacher-force the same prompt and decode token sequence through llama.cpp and ZeroMoE so autoregressive divergence cannot hide the first numerical mismatch. Compare each layer's pre-MoE hidden state and routed expert IDs/weights. For recurrent models, include state inputs/outputs and the prefill-to-decode handoff.

#### If the comparison differs: debug in this order

1. **Freeze the experiment.** Record the exact GGUF and hash, reference and ZeroMoE commits, build/patch provenance, device, context, positions, prompt token IDs, forced decode tokens, thread count, optional features, and graph policy. Confirm both runs use the same prompt and that a full-fit ZeroMoE run passed the graph gate. Use an otherwise idle GPU and fresh trace directories so stale or appended dumps cannot be mistaken for the current run.
2. **Replay identical inputs.** Derive the forced token sequence from one reference run and feed it to both engines. Do not compare free-running text after the engines have chosen different tokens. Capture the same token positions and tensor shapes in both runs, preserving raw values and metadata.
3. **Find the earliest divergent boundary.** Compare in execution order: token IDs/embeddings; layer input and pre-attention norm; attention or GDN projections, normalization, RoPE/convolution and state; attention output and residual; post-attention norm/MoE input; router logits and probabilities; top-k IDs and normalized weights; routed gate/up/activation/down; shared expert; combine and residual; next-layer hidden state. First locate the earliest layer whose boundary differs, then add detailed taps only inside that block.
4. **Diagnose from the first mismatch.** If layer input differs, inspect the preceding layer's attention/GDN output, residual, recurrent state, and position/state handoff. If the MoE input matches but router logits do not, inspect the router projection weights, matrix orientation/layout, dtype, accumulation, and normalization. If logits match but IDs differ, inspect top-k tie-breaking, expert indexing, and order; record the margin between the last selected and first excluded experts. If IDs and weights match but expert output differs, align slots by expert ID and compare gate, up, activation, and down separately, including quantized kernel/accumulation behavior. If expert outputs match but the next hidden state differs, inspect shared-expert weighting, routed-weight application, combine order, residual order, and state/cache writes.
5. **Quantify and fix one cause at a time.** Compare tensors only after confirming shape, layout, dtype, and token row match. Report relative RMS, maximum absolute error, and cosine similarity; for routing, also report exact expert IDs/weights and top-k boundary margin. Treat a route-ID mismatch or an unexplained error as unresolved. Check the llama.cpp source operation order before changing code. Make the smallest correction at the first divergent operation, then rerun the same trace before broadening it to more layers or tokens.
6. **Re-run acceptance after parity is restored.** Repeat the teacher-forced comparison, full-fit graph gate, and matched performance gate on the final binary. Do not tune later layers or accept a speed result collected from a fallback path while correctness is unresolved.

For Qwen, `ZEROMOE_QWEN4EXP_PROBE_DUMP=<directory>` writes per-layer graph taps, including layer inputs, router logits/probabilities, and GDN probes. `ZEROMOE_QWEN4EXP_PROBE_LAYER=<layer>` adds detailed gate/up/activation/down taps for one selected MoE layer. Use `ZEROMOE_TEACHER_FORCE_TOKEN_FILE=<file>` with `ZEROMOE_TEACHER_FORCE_SKIP_SAMPLING=1` to replay the same decode tokens. Use the matching llama.cpp evaluation-callback or reference trace to capture the same boundaries. Other model families should expose equivalent adapter-specific taps rather than relying on output text alone.

### 6. Benchmark only the validated graph path

Benchmark after the full-fit graph gate and teacher-forced parity checks. Use the same GGUF, physical GPU, context, active prompt length, generated token count, and thread count in ZeroMoE and llama.cpp. For ZeroMoE `--bench-random-tg N`, decode starts from one prompt token; match it with llama-bench `-pg 1,N`. For prefill `--bench-random-pp P`, compare with llama-bench `-p P`.

Without a user-set target, steady-state decode must reach at least 90% of llama.cpp. A user-set absolute floor must also pass via `--minimum-tps`. Keep the model in progress below either applicable threshold; a graph gate or coherent output is not a performance pass.

Run the relative and absolute performance gate with the matching workload:

```sh
python3 tools/check_full_graph_moe_perf.py <zeromoe.log> <llama-bench.json> \
  --n-prompt <n> --n-gen <n> --device <device> [--minimum-tps <tps>]
```

Report prefill throughput, graph construction/upload, steady-state decode, and end-to-end request rate separately. Keep profiler runs separate from uninstrumented throughput measurements. If llama.cpp cannot provide a comparable reference, state why and establish a quantitative model-specific bar before calling onboarding complete.

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
