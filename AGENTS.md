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
- Without a user-set performance target, require steady-state decode to reach
at least 90% of llama.cpp on the same GGUF, GPU, context, prompt length, and
thread count. Report prefill, graph build/upload, steady decode, and
end-to-end rates separately. Keep the model in progress below that bar and
continue profiling; do not stop at coherent output or graph-gate success.
- Enforce the performance bar with
`python3 tools/check_full_graph_moe_perf.py <zeromoe.log> <llama-bench.json> --n-prompt <n> --n-gen <n> --device <device>`.
- Keep model-specific equations and state in the adapter; reuse `GraphCore`,
  shared routed-MoE builders, placement, cache, and offload strategies.

The authoritative design and memory-fallback contract is in
[`docs/design/full-graph-model-onboarding.md`](docs/design/full-graph-model-onboarding.md).
