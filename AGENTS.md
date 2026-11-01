# ZeroMoE agent instructions

## Large file creation

- Before creating or copying any individual file larger than 10 GiB, get the
  user's explicit permission. First state the destination and expected size;
  check available space and explain any existing files that would be replaced.
  This applies to generated model shards, spliced GGUFs, build artifacts, and
  other large outputs. Do not treat available disk space as permission.

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
Comparator tools are correctness gates: they must exit nonzero at the first mismatching boundary and save a compact repro with the exact prompt/forced tokens. A successful process exit, nonempty report, or coherent output is not a parity pass. Zero generated tokens or zero graph replays is a no-op and must fail the run.
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
`python3 tools/check_full_graph_moe_perf.py <zeromoe.log> <llama-bench.json> --n-prompt <n> --n-gen <n> --context <ctx> --n-threads <n> --n-gpu-layers <n> --n-cpu-moe <n> --device <device>`.
- Compare the exact decode workload: same GGUF, physical GPU, prompt and
  generated-token counts, context, effective graph CPU threads, and expert
  placement. The checker validates context/threads from the ZeroMoE log and
  GPU-layer/CPU-MoE/device/model fields against the llama-bench row.
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
- Set and record the profile-specific absolute `--minimum-tps` before the
  first matched performance run. Apply it to the first valid rate sample;
  do not wait until the end of onboarding to discover the target was missed.
- Before any long performance run on a new model, compare its effective
  configuration with the nearest accepted profile using
  `tools/compare_full_graph_profiles.py` and actual CLI/harness overrides.
  Pass reference CLI/harness values with `--reference-override` and
  candidate effective values from startup logs or hardware policy with
  `--candidate-override`. These compare inputs do not replace live memory
  planning or placement logs.
  Save the report and review CPU-affinity layers, requested and planner-selected
  device layers, graph CPU threads, expert hotset bytes/coverage, DDR preload,
  and pinning. Treat unexplained differences as a profile-transfer failure;
  do not spend a long run on a nominal layer count that resolves differently.
- Keep model-specific equations and state in the adapter; reuse `GraphCore`,
  shared routed-MoE builders, placement, cache, and offload strategies.
- For every distinct `(gate, up, down)` expert quantization tuple in the GGUF,
  record which CPU/GPU/fused path actually executes it. A generic
  `[QUANT] ... GPU-capable` line is not evidence that a fused cold-expert path
  accepts the CPU dot type. Exercise each tuple with bit-exact fused-versus-
  stock checks, and log which tuples select the fused path or its fallback.

The authoritative design and memory-fallback contract is in
[`docs/design/full-graph-model-onboarding.md`](docs/design/full-graph-model-onboarding.md).

## Cross-session work and benchmark provenance

- Before selecting a source tree or binary, record `git status --short
  --branch`, `git rev-parse HEAD`, and `git worktree list --porcelain`. A user
  report that work was merged makes current master authoritative; a detached
  worktree or an executable with matching MTP files is not proof that the full
  runtime/configuration is current.
- Before a performance run, validate the executable's build manifest with
  `python3 tools/build_provenance.py check --binary build/zeromoe --repo .`.
  If the manifest is absent or its source commit/hash differs, rebuild from the
  selected worktree or label the run exploratory. Never report a stale binary
  as a current-master result.
- Save each run's exact TOML, prompt/token count, command/environment, stdout,
  engine log, binary manifest, and raw timing boundary under a persistent
  `build_mtp_results/<date>-<source>-<workload>/` directory. Do not leave the
  only copy in `/tmp` or another tmpfs.
- Update the versioned shared handoff at
  `docs/handoff/qwen4exp-mtp-current.md` after a merge or benchmark. Separate
  locally measured results from user/report/source-only figures; include the
  source/worktree owner, next action, and artifact paths.
- Treat other worktrees' uncommitted files as owned by their session. Inspect
  them read-only and preserve them; do not reset, overwrite, or silently mix
  their changes into a benchmark build. If such changes affect the target
  family, snapshot their exact diff hash and label the build as that snapshot.


## Required timing attribution

- Qwen timing callbacks that visit every GGML node are execution-invasive:
  they can split a multi-node CUDA subgraph and disable fused top-k and
  weighted-MoE reduction kernels. `log_level = "DEBUG"` alone must not enable
  them. Keep `ZEROMOE_QWEN4EXP_LAYER_TIMING` and
  `ZEROMOE_QWEN4EXP_LAYER_TIMING_VERBOSE` unset for correctness baselines and
  performance acceptance runs. Enable them only in a separate diagnostic run,
  record the fusion matcher results, and label its timing as instrumented.
- Intermediate output taps can also change fusion when they materialize an
  input to a fused operation. Compare normal routed outputs separately from
  materialized per-op boundary traces; never treat those traces as the normal
  execution path without an A/B that confirms the same kernels were selected.
- For every model performance investigation, record four timing views: per-layer/per-op, GPU device work, CPU work, and request workflow. Split CPU execution from CPU scheduling/barrier/wait time. Measure CUDA kernel or graph-node duration on the device; an asynchronous scheduler callback only measures host submission. Include expert hot/cold route counts where they explain a layer's cost. Keep instrumented profiling separate from the uninstrumented throughput run. Missing GPU or per-layer attribution must be stated as a blocker to root-cause claims, not filled with an estimate.
