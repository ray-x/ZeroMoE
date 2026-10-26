# Full graph model design and onboarding

## Required CUDA MoE contract

CUDA MoE support is full-graph support. A family with routed experts must have
a registered `FullGraphModel` adapter before it can run on CUDA. A CUDA MoE
family with no adapter is unsupported and startup fails; adding kernels to the
legacy layer-by-layer executor is not an onboarding path.

The adapter's `plan_decode_memory()` is the only authority that can permit
layered decode. `Fits` requires graph decode; graph disablement, decode graph
build failure, or execution failure is fatal. `Unknown` is fail-closed and also
requires graph decode. Layered decode fallback is permitted only for an
`Insufficient` result with required bytes greater than available bytes and a
human-readable reason. The planner must include the graph's weights, KV/recurrent
state, and workspace, not just the GGUF file size or MoE cache. A graph-only
adapter cannot authorize a layered fallback. `Unsupported` means the adapter's
architecture or tensor-topology check rejected the GGUF; startup fails before
large graph allocations, and this result never means memory fallback.

For a full-fit onboarding run, use:

```sh
python3 tools/check_full_graph_moe_run.py run.log --full-fit
```

For Qwen3.5/Qwen3.6 full-fit runs, add
`--require-all-expert-layers-on-device`. This also checks that the placement
proof reports every expert layer and a nonzero full expert count per layer.

The gate requires an enabled decode adapter, a `Fits` memory plan, at least one
successful graph decode, and zero layered decode tokens or legacy per-expert
dispatches. If the planner proves the graph cannot fit, validate the explicit
fallback separately with `--allow-memory-fallback` and its logged reason.

Coherent output is only a smoke check; it is never an onboarding acceptance
criterion. Before marking a model supported, teacher-force the same token
sequence through llama.cpp and ZeroMoE, compare pre-layer hidden states and
per-layer routed expert IDs/weights, and record decode correctness evidence.

If the user has not set a model-specific performance target, the default
steady-state decode bar is at least 90% of llama.cpp on the same GGUF, GPU,
context, prompt length, and thread count. Record prompt prefill, one-time graph
build/upload, steady-state decode, and end-to-end request rates separately.
If the baseline is below 90%, keep the model in progress and continue profiling
and optimization; do not stop at the first coherent token or a passing
full-graph execution gate. If llama.cpp cannot provide a comparable baseline,
record the limitation and establish a quantitative model-specific bar before
calling onboarding complete.

For a reproducible combined prompt/decode baseline, save `llama-bench` JSON
for the same GGUF and device, then run:

```sh
python3 tools/check_full_graph_moe_perf.py run.log llama-bench.json \
  --n-prompt 16 --n-gen 400 --device CUDA1
```

The performance gate uses graph-run time only, so graph construction and
upload remain separately visible in the ZeroMoE log.

## Execution path

1. `GGUFFile::load_header` reads model metadata and tensor locations. `detect_model_family` in `models/base/model_family.hpp` selects a family.
2. `GGUFGraphInitializer::build_model_graph` in `models/base/graph_initializer.cpp` derives the architecture, validates family topology, resolves `FamilyOps`, allocates state caches, and builds the dense weight manifest.
3. `FullGraphModel` in `models/graph/full_graph_model.hpp` owns the graph lifecycle: enable, bootstrap, build, run, and shutdown. `FullGraphModelBase` centralizes request validation and prefill/decode dispatch. `full_graph_model()` in `models/graph/full_graph_models.cpp` is the registry boundary.
4. The engine reads graph capabilities and runtime tuning before cache planning and preload. It then plans expert ownership and cache sizes. The generation loop asks the adapter to build and run prompt batches or one-token decode graphs. Graph builders bind ggml tensors, schedule GPU and CPU work, and connect routed experts to cache or offload providers. Shared runtime machinery belongs under `models/graph`.

A family whose output depends on recurrent state must carry that state across prefill and decode. Cache ownership, expert placement, host-memory requirements, CPU-lane support, and the decode memory plan must be declared by the adapter. A failed decode graph build or run does not authorize layered decode: only the pre-run memory plan can do that.

## Adding a family

1. Inspect the GGUF architecture, metadata, tensor names, and auxiliary blocks. Compare tensor equations with the reference implementation. Model branding or lineage does not determine the executable architecture.
2. Trace the detected family through `resolve_family_ops()` to `full_graph_model()`. If the family has no graph adapter, implement one before adding CUDA support; do not continue through the generic layer dispatcher.
   `ModelFamily` may group multiple `general.architecture` identifiers. The adapter must explicitly whitelist each supported identifier in both its capability and memory-planner paths; registering one architecture must not imply support for the others.
3. Add exact family detection and a topology validator. Check trunk versus auxiliary layer counts, per-layer attention kind, dimensions, required tensor shapes, and quant types before allocating large caches.
4. Reuse `GraphCore`, shared routing, cache strategies, layer-layout helpers, and common state handoff where the model equations permit it. Keep model-specific validation and equations beside that model.
5. Implement graph state, weight binding, graph construction, cache handoff, expert placement, prefill, decode, and shutdown. Preserve reference operation order, residual paths, normalization, router behavior, and gate activations.
6. Register a `FullGraphModelBase` adapter. Explicitly declare its decode memory plan, device expert layer count and placement direction, CPU-lane support, host DDR requirements, RAM pin defaults, cache ownership, and prefill limits. A `Fits` plan must report positive required and available device bytes with required <= available. An `Insufficient` plan must report required > available and a reason. Leave the plan `Unknown` only while graph execution remains mandatory; unknown never authorizes fallback.
7. Validate continuation, context reset, cache bounds, and state handoff. Compare layer outputs or logits with the reference implementation before tuning throughput. Record prefill and decode separately, including prompt shape, resident expert range, host-memory mode, and CPU/CUDA graph node counts. A CUDA MoE onboarding is incomplete until `python3 tools/check_full_graph_moe_run.py <run.log> --full-fit` passes on hardware where the full graph fits. The gate requires an enabled graph, a successful decode replay, and zero one-token layered forwards or legacy expert dispatches.

## Runtime performance contract

A graph can build and generate while still being unusably slow. In one hybrid MoE first-run profile, zero resident expert layers and pageable host arenas led to about 0.64 tok/s prefill. With a nonzero expert placement request, the allocator fit 19 complete layers in VRAM and prefill reached about 20.6 tok/s. A supported CPU lane was also left disabled by the adapter. These settings were not rejected by the old design because graph lifecycle support and performance setup were separate, and placement/CPU-lane methods silently inherited zero/off defaults.

`FullGraphModel` now requires each adapter to implement `device_expert_layers()`, `device_expert_trailing()`, `supports_cpu_lane()`, and `runtime_tuning()`. This makes omissions compile errors. An empty `GraphRuntimeTuning` is an explicit declaration that no model-specific DDR preload or pinning requirement applies.

Runtime tuning is applied after GGUF discovery and before cache planning or preload. It can require complete DDR expert arenas, supply defaults for RAM pin fraction and budget, reject an explicit incompatible preload setting, and warn about explicit values that leave most host arenas pageable or provide no pin budget. The placement count is a request; the allocator clamps it to available VRAM. Startup diagnostics report adapter capabilities, tuning policy, and the resolved arena layer range. A successful graph build is not a performance pass criterion.

## Shared-design improvements

`FullGraphModelBase` centralizes request validation and prefill/decode dispatch, reducing repeated adapter code. `GraphRuntimeTuning` places model-specific resource requirements next to the graph adapter while leaving resolution and application in the shared engine setup. The common startup path now applies those settings before sizing caches, so preload and placement use one resolved configuration. Startup capability logs expose the values that previously could remain implicit. The onboarding checklist makes performance setup part of adding a model rather than a follow-up optimization task.
