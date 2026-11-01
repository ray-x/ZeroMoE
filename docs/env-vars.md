# Environment variables

Prefer the model recipe and canonical root `graph_*` TOML keys for migrated
graph settings. See [configuration and precedence](configuration.md), or run
`zeromoe --help-settings` for effects, actual family defaults and aliases.
Legacy aliases still work and override TOML, but now log migration warnings.
Many experimental controls remain environment-only; this is a staged migration.

This page lists the `ZEROMOE_*` settings a user is expected to see: the ones the README and the shipped configs
refer to, the model-path fallback, and the model-specific and diagnostic knobs that the model guides name. Deep
experiment knobs that no guide or public config mentions are not listed
here; the comprehensive source-linked inventory is in the
[configuration-knob audit](design/configuration-knob-audit.md), with its
companion JSON inventory.

Legacy readers differ; do not treat environment values as TOML:

- **Flags:** `flag_on` requires a leading `1`; `flag_unless_off` is enabled unless the
  value starts with `0`; other readers require exactly `"1"` or accept nonempty
  nonzero strings. The relevant reader determines the result, not a common parser.
- **Integers:** some readers use a leading number (`10x` becomes 10), while strict
  readers reject it and retain the prior value. Preserve existing scripts, but use
  typed TOML values when migrating.

Settings are read when the process starts, and graph settings are cached per model family. Set them before launching
`zeromoe`. Changing them while the process runs has no effect.

## Build options (CMake `-D`, not environment variables)

| Option | Default | Effect |
| --- | --- | --- |
| `GGML_CUDA` | ON outside macOS; OFF on macOS | llama.cpp/GGML build option. Controls whether ZeroMoE builds the CUDA engine; the CPU preset sets it OFF. |
| `ZEROMOE_METAL` | ON on Apple Silicon | Dispatch prefill-sized F32 matmuls (`S>=32`) to Metal/MPS in the CPU backend. Ignored elsewhere. Decode stays on NEON. |
| `GGML_AVX512_VNNI` | ON on ZeroMoE x86 builds | llama.cpp/GGML option shared by ZeroMoE and ggml CPU kernels. Set OFF on CPUs without VNNI. |
| `GGML_OPENMP` | ON | llama.cpp/GGML CPU backend OpenMP setting. OFF selects ggml's persistent threadpool. |
| `ZEROMOE_GGML_COMMIT` | `8baabd53b…` | llama.cpp commit whose `ggml/` directory is built. Changing it changes the numerics; re-validate. |
| `ZEROMOE_GGML_ROOT` | empty | Use a local ggml source tree instead of the submodule. |

## Model location

| Variable | Default | Effect |
| --- | --- | --- |

## API authentication

| Variable | Default | Effect |
| --- | --- | --- |
| `ZEROMOE_API_KEY` | unset | Bearer/API-key secret for non-loopback API serving. The CLI equivalent is `--api-key`. The variable keeps the ZeroMoE project namespace; do not log its value. |

## Qwen 3.8 Flash full graph

The shipped Qwen recipe no longer requires full-graph launch flags. Prefill and
decode default to the graph; placement uses TOML `graph_device_expert_layers` /
`graph_device_expert_trailing` or CLI `--graph-device-expert-layers` /
`--graph-device-expert-trailing`. The legacy preload environment aliases were removed.

| Variable | Set to | Effect |
| --- | --- | --- |
| ~~`ZEROMOE_QWEN4EXP_FULL_GRAPH`~~ | removed | Full-graph prefill is the default. TOML `qwen4exp_full_graph = false` is the only opt-out; set the decode key false too for a fully layered fallback. |
| ~~`ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE`~~ | removed | Full-graph decode is the default. TOML `qwen4exp_full_graph_decode = false` keeps graph prefill and selects layered decode. |
| `ZEROMOE_QWEN4EXP_PREFILL_OP_OFFLOAD` | `1` (default) | Run host-resident experts on the GPU during prefill, by copying the batch's experts across. `0` keeps them on the CPU. |
| `ZEROMOE_QWEN_CPU_THREADS` | all OpenMP threads | Positive integer. Caps (never raises) the threads used for Qwen CPU expert work. |
| `ZEROMOE_QWEN4EXP_HOST_NODE` | off | Use the host-node CPU/GPU schedule (`flag_on`). Measured in [runtime-settings.md](runtime-settings.md). |
| `ZEROMOE_QWEN4EXP_LAYER_TIMING` | off | Enable CPU node timing and dispatch/barrier counters. It installs an evaluation callback that can split CUDA graph work and disable fused kernels; keep it off for correctness and performance baselines. |
| `ZEROMOE_QWEN4EXP_LAYER_TIMING_VERBOSE` | off | Dump every scheduler node and backend for the first decode token. It is also invasive and can disable fused router/reduction kernels. Enable only for a separate diagnostic run, with `log_level = "DEBUG"` to see detailed records. `log_level = "DEBUG"` alone does not enable either timing callback. |
| `ZEROMOE_QWEN4EXP_COLD_FUSED_TIMING` | off | Aggregate per-layer wall time, task count, and cold-route counts inside the fused CPU custom node. It does not install the scheduler evaluation callback or split CUDA graphs. Diagnostic only; do not use the timed run for throughput. |
| `ZEROMOE_QWEN4EXP_PROBE_ROUTED_ONLY`, `ZEROMOE_QWEN4EXP_PROBE_ROUTED_ONLY_DIR` | off / unset | Test-only capture of final routed outputs without enabling the full intermediate probe set. Set `ZEROMOE_QWEN4EXP_PROBE_LAYER` to one layer; `-1` captures every layer. |
| `ZEROMOE_QWEN4EXP_PROBE_DUMP_LAYER` | unset (all layers) | With `ZEROMOE_QWEN4EXP_PROBE_DUMP` enabled, retain intermediate and detailed expert diagnostic roots for only this layer. It overrides `ZEROMOE_QWEN4EXP_PROBE_LAYER` for the full-dump scope. Use for correctness isolation; graph probes can affect scheduling, so do not use their runs for performance measurements. |
| `ZEROMOE_FULL_GRAPH_PROBE_DIR`, `ZEROMOE_FULL_GRAPH_PROBE_LAYER` | unset / all layers | Shared `GraphCore` boundary capture. Adapters register existing layer tensors; the writer saves raw values plus dtype/shape/stride/backend metadata. Diagnostic graph roots may affect scheduling; do not use these runs for performance. |
| `ZEROMOE_CUDA_SCHEDULE` | `auto` (`blocking` when `ZEROMOE_QWEN4EXP_HOST_NODE=1`) | CUDA synchronisation: `auto`, `blocking` or `spin`. Other values select the default. |

### Qwen memory policy

| Variable | Default | Effect |
| --- | --- | --- |
| `ZEROMOE_QWEN_GRAPH_REGISTER` | on for Qwen | Pin (register) the host expert arenas. Pinned copies run at about 10.6 GB/s against about 4.9 GB/s pageable, which cuts the per-request decode setup from about 3.9 s to 2.0 s. Set `0` to keep the arenas pageable, for machines where the pinned budget is too small. `ZEROMOE_QWEN_GRAPH_NO_REGISTER=1` also disables it. |
| `ZEROMOE_QWEN_SOFT_PIN_GIB` | `6` | Size, in GiB, of the expert tier that a prefill can release. Only that tier is re-uploaded after the prefill, so a second request's setup is about 0.4–0.7 s for 700–1,700-token prompts. `0` disables the tier. The tier is capped at half the split payload. |
| `ZEROMOE_MEM_GUARD_FLOOR_MB` | 20480 | Lower the MemAvailable stop floor (in MB) during long prefills. Used by the 74 GiB pool configurations. |
| `ZEROMOE_DENSE_SCRATCH_RESERVE_MB` | `256` | Reserve, in MB, for dense-layer scratch memory before the VRAM budget checks run. Non-positive or malformed values fall back to 256. Read in `core/attention_layer.cpp`. |

### Qwen early-ready lane (experimental)

The Qwen3.8 decode runner sets ALL_SPLIT=1, LAYER13=0, and
QWEN_CPU_THREADS=10. ASYNC_OUTPUT is left unset, so the adapter default is
used. The marker must be in the forward graph for the scheduler to run the CPU
lane. Current Qwen graph construction roots that marker explicitly.

| Variable | Meaning and default |
|---|---|
| ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT | Opt in to early-ready CPU lanes on every routed hot/cold split. Unset or 0 uses the regular split path. |
| ZEROMOE_QWEN_EARLY_READY_LAYER13 | Opt in to layer 13 only when all-split is not enabled. |
| ZEROMOE_QWEN_EARLY_READY_ASYNC_OUTPUT | Architecture default: on for `qwen4exp`, off for `qwen35moe`. Explicit 0 selects scheduler-managed synchronous output; explicit 1 selects the side-stream output copy. The defaults reflect their separate matched profiles; compare the exact architecture before changing them. |
| ZEROMOE_EARLY_READY_ROUTE_BUNDLE_TRACE | Default off. When set to 1, logs top-k expert IDs and normalized weights per early-ready lane call for usage-survey diagnostics. Adds logging overhead; do not use for performance measurements. The terminal MoE layer without an early-ready lane is not included. |
| ZEROMOE_QWEN_EARLY_READY_RESIDENT_TABLES | Keep immutable route tables resident on GPU; default on, explicit 0 disables. |
| ZEROMOE_QWEN_EARLY_READY_PREDICT | Enable next-layer route prediction. Default off unless set to 1. |
| ZEROMOE_QWEN_EARLY_READY_PREDICT_LAYER | Restrict prediction to one target layer; unset predicts the next split layer. |
| ZEROMOE_QWEN_EARLY_READY_RIDGE | Path to an optional ridge correction table; missing or invalid files leave the plain predictor. |
| ZEROMOE_QWEN_EARLY_READY_SKIP_ALL_HOT | Skip the CPU lane when all selected experts are resident; enable only after correctness/performance checks. |
| ZEROMOE_QWEN_EARLY_READY_STAGE_LAYER, ZEROMOE_QWEN_EARLY_READY_STAGE_VERIFY | Select layer(s) for staged expert copies and route verification. |
| ZEROMOE_QWEN_EARLY_READY_ROUTE_TRACE | Diagnostic path for lane route decisions. |

Timing callbacks and probes change graph partitioning and are not performance
settings. See the [scheduler overlap worklog](option-b-graph-refactor-steps-2026-10-04.md)
and the [Qwen3.8-to-Ornith profile audit](handoff/qwen38-to-ornith-profile-transfer-2026-10-31.md).

## Expert offload (SSD and host memory)

| Variable | Default | Effect |
| --- | --- | --- |
| `ZEROMOE_OFFLOAD_RETARGET` | on | Retarget offloaded expert reads when SSD offload is active. `0` disables. See [offload-retarget.md](offload-retarget.md). |
| `ZEROMOE_OFFLOAD_EVICT` | `MADV_DONTNEED` | Set to `free` to evict with `MADV_FREE` instead: pages stay mapped until the kernel reclaims them, so the next read skips the page fault, at the price of a soft memory budget. Any other value (or unset) uses `MADV_DONTNEED`. |

## GLM-5 / Ling / MiMo / DeepSeek families

These are model-specific. The model's own document (`docs/models/<model>.md`) is the authority for its values.

| Variable | Default | Effect |
| --- | --- | --- |
| ~~`ZEROMOE_GLM_DSA_FULL_GRAPH`~~, ~~`ZEROMOE_LING_FULL_GRAPH`~~, ~~`ZEROMOE_DEEPSEEK_V41_FULL_GRAPH`~~ | removed | Full graph is the default for these families and cannot be disabled through the environment. `ZEROMOE_LING_NATIVE_GRAPH` still selects Ling's native cache-backed graph; `ZEROMOE_DEEPSEEK_V41_FULL_GRAPH_DECODE=0` remains an explicit native-decode comparison. |
| `ZEROMOE_GLM_DSA_CPU_LANE` | off | Advertise the GLM DSA CPU lane: cold experts may run on a private CPU lane concurrently with the layer's GPU work (the early-ready/gpu-assist capability). Only a leading `1` enables it; without it, a mixed VRAM+DDR placement logs a `PLACEMENT-WARN`. |
| `ZEROMOE_GLM5NEXT_CPU_THREADS` | `6` | ggml CPU threads for GLM-5 Next. Prefill prefers `ZEROMOE_GLM5NEXT_PREFILL_CPU_THREADS` when that is set. |
| `ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB` | `2048` | Free-VRAM floor the graph allocator must leave before it builds GLM-5 Next graphs, in MiB. `0` removes the floor; it also gates pilot staging. |
| `ZEROMOE_GLM5NEXT_GPU1_HOT_GIB` | unset | Hot-expert budget on the second GPU, in GiB (multi-GPU runs only). Unset or empty leaves the GPU1 hot set unconfigured. |
| `ZEROMOE_GLM5NEXT_GPU1_HOT_PRELOAD` | `1` | Preload the second-GPU hot set. `0` disables. |
| `ZEROMOE_GLM5NEXT_PREFILL_GPU1` | `1` | Use the second GPU during prefill. `0` disables. |
| `graph_*` settings | Typed TOML keys and generic `--graph-*` CLI flags. | Offload and scheduler settings use canonical names such as `--graph-prefill-op-offload 1`, `--graph-pilot-copy 0`, and `--graph-offload-split-reads 1`; legacy GLM5Next env aliases are retired. |
| `ZEROMOE_GLM5NEXT_PILOT_PREFETCH` | `0` | Pilot prefetch depth, clamped to 0–8; above `0` enables the pilot. The effective depth is at least `ZEROMOE_GLM5NEXT_PILOT_STAGE`, and predicted experts that are not in host memory are read from SSD ahead of the layer that needs them. |
| `ZEROMOE_GLM5NEXT_PILOT_STAGE` | `0` | Number of next-layer cold experts the pilot stages into the VRAM ring per layer, clamped to 0–8. `0` disables pilot staging. |
| `ZEROMOE_GLM5NEXT_MODEL_DIRECT` | off | Read model tensors directly (`1` enables). |

## Benchmark, diagnostic and test

| Variable | Effect |
| --- | --- |
| `ZEROMOE_BENCH_IGNORE_STOP` | Benchmark runs ignore end-of-sequence tokens, so every run generates its full token count. Set to `1` in bench scripts. |
| `ZEROMOE_BENCH_SILENT` | Suppress per-token printing and per-token VRAM sampling in the timed path. |
| `ZEROMOE_QWEN4EXP_MTP_WINDOW=0` | Force the graph-native Qwen MTP target to verify drafted tokens stepwise instead of in a multi-token window. ZeroMoE-specific correctness comparison; default uses the window derived from `spec_draft_n_max`. |
| `ZEROMOE_GENERATED_TOKEN_FILE` | Write each generated token id to this file, one per line (truncated at step 0). For numerical comparisons. |
| `ZEROMOE_DECODE_LOGITS_DUMP` | Dump decode logits to this file. Diagnostic. |
| `ZEROMOE_QWEN4EXP_DIAG_DUMP_DIR` | With `--verbosity 5`, write layer-0 Qwen CPU graph tensor snapshots into this directory. |
| `--verbosity 5` | Emits Qwen layer-0 CPU graph diagnostics (`[QWEN4EXP-LAYER0-DIAG]`); the separate dump-directory variable controls binary snapshots. |
| `ZEROMOE_WUV_INT8` | Attention value-projection int8 path (`1` enables). Diagnostic; read in `core/attention_layer.cpp`. |
| `ZEROMOE_TEST_EXL3_CPU_ONLY`, `ZEROMOE_TEST_EXL3_ONLY` | Stop the native test harness early: after the CPU/EXL3 checks with `_CPU_ONLY`, after the CUDA kernel checks with `_ONLY` (the form `README.md` uses for the short CUDA check). |
