# ZeroMoE agent instructions

## CUDA MoE architecture is full graph

- Do not add a new CUDA MoE family through the legacy layer-by-layer expert
  executor. CUDA MoE support requires a registered `FullGraphModel` adapter
  built on the shared graph runtime.
- Before implementing a family, trace `general.architecture` through
  `detect_model_family()`, `resolve_family_ops()`, and `full_graph_model()`.
  If no graph adapter is registered, implement the adapter before claiming
  CUDA support; the runtime must reject the family in the meantime.
- `ModelFamily` can group several GGUF architecture identifiers. An adapter
  must whitelist the exact `general.architecture` values it implements in its
  capability and planner paths; a shared family label is not proof of model
  compatibility.
- A decode graph memory plan is mandatory evidence. `Fits` requires positive
  required and available device-byte estimates with required <= available.
  `Unknown` does not authorize layered fallback. Only `Insufficient` with
  required > available and a logged reason permits it; graph-only models cannot
  fall back.
- An adapter that rejects the exact GGUF architecture or tensor topology must
  report `Unsupported`, which fails startup before large graph allocation and
  never authorizes a memory fallback.
- If the planner says the graph fits, graph decode must be enabled and used.
  A graph build or run failure is fatal for that request. Never silently route
  into layered decode.
- Do not mark a full-fit CUDA MoE onboarding complete until
  `python3 tools/check_full_graph_moe_run.py <run.log> --full-fit` passes. It
  checks graph capability, successful decode replays, and zero one-token
  layered forwards or legacy per-expert dispatches.
- A coherent sample is only a smoke check, never onboarding acceptance.
Compare teacher-forced pre-layer hidden states and routed expert IDs/weights
with llama.cpp before accepting correctness.
- Treat llama.cpp's exact operation sequence as the initial executable
  specification, not just its algebraic equations. Record the reference source
  commit/path and map each block stage to the ZeroMoE tensor/op boundary,
  including shape/layout, dtype, norm function and epsilon, scale placement,
  activation, residual order, and state handoff. Do not substitute a
  mathematically equivalent GGML sequence or fuse/reorder operations without a
  same-input boundary-trace A/B; rounding changes can amplify in quantized MoE
  and change near-tie routing.
- Without a user-set performance target, require steady-state decode to reach
at least 90% of llama.cpp on the same GGUF, GPU, context, prompt length, and
thread count. Report prefill, graph build/upload, steady decode, and
end-to-end rates separately. Keep the model in progress below that bar and
continue profiling; do not stop at coherent output or graph-gate success.
- Enforce the performance bar with
`python3 tools/check_full_graph_moe_perf.py <zeromoe.log> <llama-bench.json> --n-prompt <n> --n-gen <n> --device <device>`.
- Compare the exact decode workload: same GGUF, GPU, active prompt length,
  generated token count, context, and thread count. The perf checker verifies
  the ZeroMoE decode request's prompt length against the llama-bench row.
- Record both project commits, the shared upstream GGML base revision/version,
  ZeroMoE's applied GGML patch manifest, CUDA toolkit and driver, compiler, and
  native GPU code targets. Project-only patches are part of the implementation
  being measured; do not describe the runs as kernel-identical. If the upstream
  base, toolchain, or GPU target differs, label the ratio provisional and
  rebuild a matched reference before accepting performance.
- In a copied GGML build tree, verify the generated `GGML_COMMIT` reports the
  pinned/upstream GGML revision, not the enclosing ZeroMoE repository commit.
- Enforce a user-set absolute decode target with `--minimum-tps` in addition
  to the default relative llama.cpp ratio. A model remains in progress until
  both applicable thresholds pass on repeatable steady-state measurements.
- Keep model-specific equations and state in the adapter; reuse `GraphCore`,
  shared routed-MoE builders, placement, cache, and offload strategies.

The authoritative design and memory-fallback contract is in
[`docs/design/full-graph-model-onboarding.md`](docs/design/full-graph-model-onboarding.md).
