# Environment surface: full-graph defaults and Qwen preload (2026-10-27)

## Status

- Code commit: `a8a1c7d` on `master`; this report is the follow-up docs commit.
- `refactor/env-surface` is fast-forwarded to the same history.
- Validation build: `build_verify_master/` (CUDA, arch 86). The provenance
  manifest records `models/base/family_ops.cpp` as an existing uncommitted
  file owned by another session; it is not part of this change. Its diff
  snapshot is
  `sha256:4529977cca6ef2912956955fa52e0b09a7c6d2f62515ba50d2b7294d48d50e28`
  (1 insertion, 2 deletions). The build is labelled with that snapshot.
- No files in `ZeroMoE-wt-mtp`, `ZeroMoE-wt-mtp2` or
  `build_mtp_results/worktree_qwen_mtp` were modified.

## Behavior after this change

| Family | Prefill graph | Decode graph | Explicit fallback / selector |
| --- | --- | --- | --- |
| Qwen4Exp | on by default | on by default | TOML is the only opt-out. `qwen4exp_full_graph_decode=false` alone keeps graph prefill and selects layered decode; to select layered prefill as well, set both keys false (prefill false with decode still true is rejected as an inconsistent request). |
| GLM-DSA | on by default | on by default | `engine.mtp_enabled` keeps the layered path; the adapter has no speculative graph decode. |
| Ling 3.1 Flash | on by default | on by default | `ZEROMOE_LING_NATIVE_GRAPH` selects the cache-backed native graph; without it the shared GLM5Next graph is used. `ZEROMOE_LING_HYBRID_CUDA_GRAPH` still selects the hybrid implementation. |
| DeepSeek-V4.1 | on | on by default | `ZEROMOE_DEEPSEEK_V41_FULL_GRAPH_DECODE=0` selects native decode while graph prefill stays on. `engine.mtp_enabled` also opts out. |
| GLM5Next | on by default (existing) | on by default (existing) | `ZEROMOE_GLM5NEXT_FULL_GRAPH_PREFILL=1` remains the prefill opt-in for the CUDA MMQ route-compaction limitation; not changed here. |

The CPU-only build keeps the graph stubs returning `false`. The graph needs
the CUDA backend; "always on" means "always the default whenever the family's
CUDA graph is compiled and selectable", not a CPU execution path.

## Removed environment names

| Removed name | Replacement |
| --- | --- |
| `ZEROMOE_QWEN4EXP_FULL_GRAPH` | Qwen full-graph prefill defaults on; TOML `qwen4exp_full_graph` is the only opt-out. |
| `ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE` | Qwen full-graph decode defaults on with prefill; TOML `qwen4exp_full_graph_decode` is the only opt-out. |
| `ZEROMOE_GLM_DSA_FULL_GRAPH` | GLM-DSA graph is the default. |
| `ZEROMOE_GLM_DSA_FULL_GRAPH_PREFILL` | GLM-DSA batched prefill graph is the default. |
| `ZEROMOE_LING_FULL_GRAPH` | Ling full graph is the default; `ZEROMOE_LING_NATIVE_GRAPH` selects the implementation. |
| `ZEROMOE_DEEPSEEK_V41_FULL_GRAPH` | DeepSeek-V4.1 prefill graph is the default; the decode override remains. |

The Qwen preload aliases were kept, with changed defaults:

- `ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS` (legacy alias for
  `graph_device_expert_layers`): default `2`. `0` is the explicit
  "no whole-layer block" sentinel used by the global-pool recipe. An enabled
  block must be at least 2 layers: env `1` is raised to 2 with a warning, TOML
  `1` is rejected by the typed resolver.
- `ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING` (legacy alias for
  `graph_device_expert_trailing`): default `false`. It only affects placement
  when the layer count is nonzero.

## Why `PRELOAD_TRAILING` exists

It is the placement side of `device_expert_layers`, not an independent
feature. ZeroMoE historically put the whole-layer GPU-resident block at the
start of the network; llama.cpp's `-ncmoe N` puts the same count at the end.
The flag selects between those two conventions. It has no effect when
`device_expert_layers` is 0, which is why the old documentation's `1` default
was misleading: the code default was already `false`, and most shipped
recipes either used 0 layers or set the flag only for a measured trailing
recipe.

With at least 2 layers now the default, the flag is still useful for A/B
comparison and for loading a llama.cpp-style trailing block, but it should
not be treated as a required setup switch.

## Validation

Run on this Linux host (CUDA 13.3, driver 610.57.04):

```sh
# CUDA build
cmake -S . -B build_verify_master -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build_verify_master --target zeromoe -j"$(nproc)"

# settings tests
cmake --build build_verify_master --target \
  graph_settings_test graph_settings_binding_test -j"$(nproc)"
./build_verify_master/graph_settings_test tests/graph_settings_defaults.snapshot
cd build_verify_master && ctest -R '^graph_settings' --output-on-failure

# CPU-only configure/build
cmake -S . -B /tmp/opencode/envsurface-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DZEROMOE_CPU_ONLY=ON
cmake --build /tmp/opencode/envsurface-cpu --target \
  zeromoe_cpu graph_settings_test -j"$(nproc)"

# config migration guard
python3 tools/config_inventory.py --check
```

Results:

- CUDA `zeromoe` build: success. `python3 tools/build_provenance.py check
  --binary build_verify_master/zeromoe --repo .` passes with
  `source=837ffc1a614e8cb7c98ac8ef4e4b07e998ccc12e dirty=True
  binary_sha256=5ebdc2638c8310c52adf10820adcfac12cfe4c046c312b0273c709c2e42f98e3`
  and `ggml_base=8baabd53b77d1c48c6ef8d3b620436292f3f6231`. The dirty flag
  is the `family_ops.cpp` snapshot above.
- `graph_settings` and `graph_settings_binding`: 2/2 pass.
- Defaults snapshot now contains `device_expert_layers=2`; explicit `0` is
  still accepted and TOML `1` is rejected.
- CPU-only `zeromoe_cpu` and `graph_settings_test`: build and pass.
- `tools/config_inventory.py --check`: "Migrated Qwen settings use the typed
  resolver."
- `git diff --check`: clean.
- No live code reads of the six removed environment names remain. Dated
  benchmark records (`bench/results/`, `docs/perf-record-*`,
  `docs/qwen-decode-baselines-*`, `configs/bench/perf-baselines.md`) still
  name them as historical provenance; they are not read by the engine.

## One schema for env, CLI and TOML

There is no drop-in Go Viper equivalent in C++. The closest single library is
**CLI11** (header-only, C++11): an option can carry a CLI name, an
`->envname("ZERO_CUDA_ENABLE")` alias and `->configurable()` for a config
file. **Boost.Program_options** also supports command line, environment and
config-file parsing, but it is heavier and the name mapping is manual.

This repository already contains most of the right abstraction:
`models/graph/graph_settings.hpp` has a `Field` table with the canonical TOML
name, legacy environment aliases, parser, default, family mask, and a
`Resolved<T>{value, src, origin}` that records where the value came from.
`--help-settings` is generated from the same table. The smallest safe step is
to add a CLI alias to that table and one `apply_cli()` pass, rather than add a
new dependency:

```text
canonical: zero_cuda_enable        (TOML / settings key)
env:       ZERO_CUDA_ENABLE        (legacy alias)
cli:       --zero-cuda-enable      (new alias)
precedence: CLI > env > TOML > family/default
```

Keep secrets environment-only, and keep compile-time capability macros
(`ZEROMOE_USE_GGML_CUDA`, `ZEROMOE_METAL`) separate from runtime settings:
a TOML value cannot change which translation units were compiled. A shared
schema can generate the CMake cache variable and its documentation, but it
must not pretend that a runtime key selects a build capability.

## Follow-ups

- Re-run the Qwen, GLM-DSA, Ling and DeepSeek-V4.1 full-graph hardware
  validations; the code path is default-on now, but this host did not run
  each family's model.
- Qwen default `device_expert_layers=2` changes the default VRAM placement.
  Check the `[GRAPH-CAPS] device_expert_layers=...` and
  `[GGML-QWEN-FULL] device expert arena` lines on low-VRAM devices; use an
  explicit `graph_device_expert_layers = 0` for the global-pool recipe.
- If the project adopts a single external configuration library, prefer
  CLI11 and keep the existing `Resolved` source tracking; do not migrate the
  `[[gpus]]` array and legacy family aliases in the same change.
