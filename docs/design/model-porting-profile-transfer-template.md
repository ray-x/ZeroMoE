# New model profile transfer record

Copy this file into the model's onboarding artifacts and fill it in before
choosing runtime settings or starting a performance run. Use the user's named
reference and the measured profile that best matches its workload. A family
label or fine-tune relationship justifies a starting comparison; it does not
prove that every setting transfers.

## 1. Frozen reference run

Record the exact run that anchors the port.

- Reference model and GGUF:
- Model hash and architecture identifier:
- Reference source commit and source status:
- Reference binary and hash:
- Upstream GGML base, applied patch manifest, compiler, CUDA toolkit/driver:
- Physical GPU name and UUID:
- Context, active prompt length, generated token count, batch/ubatch:
- Exact prompt/token IDs and forced decode tokens:
- Decode thread count from both configuration and startup log:
- Device placement, resident expert bytes/count, host expert source:
- Usage survey source, prompts, route observations, weights, and coverage:
- Optional features and graph policy:
- Steady decode, prefill, graph setup/upload, end-to-end rate:
- Run directory and full-graph/performance gate outputs:

Before selecting a candidate binary or source tree, capture:

    git status --short --branch
    git rev-parse HEAD
    git worktree list --porcelain

Validate the binary build provenance and preserve its complete runtime library
bundle. Keep the reference files immutable; create separate candidate configs
and logs.

## 2. Reference rationale and topology differences

- User-named reference:
- Why it is the closest measured profile:
- Verified lineage:
- Exact target architecture and tensor topology:
- Attention/state/residual differences:
- Expert count, top-k, gate/up/down shapes and quantization:
- Active routed weights/bytes per token:
- Which reference settings are expected to transfer, and why:
- Which settings remain model-specific or unverified:

Never transfer another model's usage ranking as if it were target-model data.
If a prompt distribution or survey weighting is changed to improve a benchmark
result, label the profile with that weighting and keep it separate from a
general usage profile.

## 3. Setting transfer matrix

Every performance-relevant setting must have a source and a disposition. Use
only these dispositions: Transfer, Model-specific, Unverified, or Rejected.

| Area | Reference value and source | Candidate value and source | Disposition | Evidence or reason |
|---|---|---|---|---|
| GPU identity and CUDA target | | | | |
| Dense/attention placement | | | | |
| Expert placement and VRAM byte budget | | | | |
| Host expert source, preload, pinning | | | | |
| Usage file and prompt weighting | | | | |
| CPU threads and affinity | | | | |
| Prefill batch and workspace policy | | | | |
| Early-ready/async output/CPU lane | | | | |
| Cache, prefetch, promotion | | | | |
| Fusion/kernel selection | | | | |
| Graph fallback policy | | | | |

Do not leave Unverified entries before a long performance run. Resolve them with
a short config-resolution check, a capability check, or a targeted A/B. Record
the effective value from startup logs, not just the intended TOML or shell
assignment.

## 4. Correctness gate before optimization

- Reference llama.cpp commit/path and operation sequence:
- Per-block mapping to ZeroMoE tensors/operators, including layouts, types,
  normalization/epsilon, scales, activation, residual order, and state handoff:
- Teacher-forced prompt and forced token sequence:
- Placement equivalence or unavoidable difference:
- Per-layer pre-MoE hidden-state comparison:
- Routed logits/probabilities, ordered expert IDs, selected/normalized weights:
- Recurrent-state and prefill-to-decode handoff comparison, if applicable:
- First divergent boundary and maximum/mean error:
- Fused-versus-stock result for the exact quantization and dimensions:

Stop at the first unexplained hidden-state or routed-expert mismatch. Change
one operation or execution setting at a time. Keep a fused or asynchronous path
opt-in until same-input full-graph traces pass. A coherent sample is only a
smoke check.

## 5. Full-graph and performance gates

Record the memory plan before allocating the large graph. Fits requires
positive required and available device-byte estimates with required no greater
than available. Unknown never authorizes layered fallback. If the planner says
the graph fits, build/run failures are fatal.

- Full-fit run gate:
- Strict full-graph prefill gate:
- Zero layered one-token forwards / zero legacy expert dispatches:
- Matched llama-bench row and hardware placement:
- Matched prompt tokens, generated tokens, context, and effective thread count:
- User-set absolute decode floor:
- Relative and absolute gate commands/results:
- Repetitions, sample range, and host load:

Report separately: expert preload, prefill, graph build/upload, steady graph
decode, request decode wall, and end-to-end rate. Timing must include per-layer
CPU/operation attribution, GPU kernel or graph-node attribution, CPU waits and
barriers, and request-workflow stages. If a view cannot be measured, record the
missing instrumentation and keep that cause unresolved. Never use an invasive
timing callback in the performance acceptance run.

## 6. Handoff and status

- Source worktree/branch and dirty status:
- Binary hash/build manifest and runtime library paths:
- Persistent run-artifact directory:
- Correctness status:
- Full-graph gate status:
- Performance status against both the 90% relative bar and user absolute floor:
- Accepted profile scope, including prompt weighting and hardware:
- Remaining gaps and the next falsifiable experiment:

Keep the model in progress if any applicable gate fails or any requested scope
is not verified. Mark support complete only for the explicit model, quant,
device, workload, and profile that passed every required gate.
