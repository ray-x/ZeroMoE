# Full graph model design and onboarding

## Execution path

1. `GGUFFile::load_header` reads model metadata and tensor locations. `detect_model_family` in `models/base/model_family.hpp` selects a family.
2. `GGUFGraphInitializer::build_model_graph` in `models/base/graph_initializer.cpp` derives the architecture, validates family topology, resolves `FamilyOps`, allocates state caches, and builds the dense weight manifest.
3. `FullGraphModel` in `models/graph/full_graph_model.hpp` owns the graph lifecycle: enable, bootstrap, build, run, and shutdown. `FullGraphModelBase` centralizes request validation and prefill/decode dispatch. `full_graph_model()` in `models/graph/full_graph_models.cpp` is the registry boundary.
4. The engine reads graph capabilities and runtime tuning before cache planning and preload. It then plans expert ownership and cache sizes. The generation loop asks the adapter to build and run prompt batches or one-token decode graphs. Graph builders bind ggml tensors, schedule GPU and CPU work, and connect routed experts to cache or offload providers. Shared runtime machinery belongs under `models/graph`.

A family whose output depends on recurrent state must carry that state across prefill and decode. Cache ownership, expert placement, host-memory requirements, and CPU-lane support must be declared by the adapter. If `build` can decline, the caller may fall back to layer-by-layer execution; set `graph_only()` when that path cannot reproduce the model.

## Adding a family

1. Inspect the GGUF architecture, metadata, tensor names, and auxiliary blocks. Compare tensor equations with the reference implementation. Model branding or lineage does not determine the executable architecture.
2. Add exact family detection and a topology validator. Check trunk versus auxiliary layer counts, per-layer attention kind, dimensions, required tensor shapes, and quant types before allocating large caches.
3. Reuse `GraphCore`, shared routing, cache strategies, layer-layout helpers, and common state handoff where the model equations permit it. Keep model-specific validation and equations beside that model.
4. Implement graph state, weight binding, graph construction, cache handoff, expert placement, prefill, decode, and shutdown. Preserve reference operation order, residual paths, normalization, router behavior, and gate activations.
5. Register a `FullGraphModelBase` adapter. Explicitly declare device expert layer count and placement direction, CPU-lane support, host DDR requirements, RAM pin defaults, cache ownership, prefill limits, and whether the graph is the only correct path. Do not rely on default performance values.
6. Validate continuation, context reset, cache bounds, and state handoff. Compare layer outputs or logits with the reference implementation before tuning throughput. Record prefill and decode separately, including prompt shape, resident expert range, host-memory mode, and CPU/CUDA graph node counts.

## Runtime performance contract

A graph can build and generate while still being unusably slow. In one hybrid MoE first-run profile, zero resident expert layers and pageable host arenas led to about 0.64 tok/s prefill. With a nonzero expert placement request, the allocator fit 19 complete layers in VRAM and prefill reached about 20.6 tok/s. A supported CPU lane was also left disabled by the adapter. These settings were not rejected by the old design because graph lifecycle support and performance setup were separate, and placement/CPU-lane methods silently inherited zero/off defaults.

`FullGraphModel` now requires each adapter to implement `device_expert_layers()`, `device_expert_trailing()`, `supports_cpu_lane()`, and `runtime_tuning()`. This makes omissions compile errors. An empty `GraphRuntimeTuning` is an explicit declaration that no model-specific DDR preload or pinning requirement applies.

Runtime tuning is applied after GGUF discovery and before cache planning or preload. It can require complete DDR expert arenas, supply defaults for RAM pin fraction and budget, reject an explicit incompatible preload setting, and warn about explicit values that leave most host arenas pageable or provide no pin budget. The placement count is a request; the allocator clamps it to available VRAM. Startup diagnostics report adapter capabilities, tuning policy, and the resolved arena layer range. A successful graph build is not a performance pass criterion.

## Shared-design improvements

`FullGraphModelBase` centralizes request validation and prefill/decode dispatch, reducing repeated adapter code. `GraphRuntimeTuning` places model-specific resource requirements next to the graph adapter while leaving resolution and application in the shared engine setup. The common startup path now applies those settings before sizing caches, so preload and placement use one resolved configuration. Startup capability logs expose the values that previously could remain implicit. The onboarding checklist makes performance setup part of adding a model rather than a follow-up optimization task.
