# Environment variables

Prefer the model recipe and canonical root `graph_*` TOML keys for migrated
graph settings. See [configuration and precedence](configuration.md), or run
`zeromoe --help-settings` for effects, actual family defaults and aliases.
Legacy aliases still work and override TOML, but now log migration warnings.
Many experimental controls remain environment-only; this is a staged migration.

This page lists the `ZEROMOE_*` settings a user is expected to see: the ones the README and the shipped configs
refer to, plus the model-path fallback. Internal benchmark and diagnostic knobs that no public config uses are not
listed here; they live in the source (`grep -rn ZEROMOE_ core models memory backends`).

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
| `ZEROMOE_USE_GGML` | OFF | Use ggml for the quantized CPU kernels. Required for the Qwen full graph. |
| `ZEROMOE_USE_GGML_CUDA` | OFF | Build the ggml CUDA backend for the MUL_MAT_ID prefill path (experimental). |
| `ZEROMOE_ENABLE_CUDA_GRAPHS` | ON | Enable ggml CUDA graph capture and replay. |
| `ZEROMOE_ENABLE_AVX512_VNNI` | ON | Compile with AVX-512 VNNI. Set OFF on a CPU without VNNI, or the binary fails with an illegal instruction. |
| `ZEROMOE_GGML_COMMIT` | `8baabd53b…` | llama.cpp commit whose `ggml/` directory is built. Changing it changes the numerics; re-validate. |
| `ZEROMOE_GGML_ROOT` | empty | Use a local ggml source tree instead of the submodule. |

## Model location

| Variable | Default | Effect |
| --- | --- | --- |
| `ZEROMOE_MODEL_PATH` | unset | Model file or shard directory. Used when the config's `model_path` is a placeholder (`/path/to/…` or `path_to_…`) and no `--model` is given. Precedence: `--model`, then a real `model_path` in the config, then this variable. If none is set, the engine stops with a message. |

## Qwen 3.8 Flash full graph

The shipped Qwen recipe no longer requires these variables. Full-graph prefill/decode
default to enabled; placement uses `graph_device_expert_layers` and
`graph_device_expert_trailing`. The table retains the historical launch values for
existing scripts.

| Variable | Set to | Effect |
| --- | --- | --- |
| `ZEROMOE_QWEN4EXP_FULL_GRAPH` | `1` | Run prefill through the full ggml graph. |
| `ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE` | `1` | Run decode through the full graph. |
| `ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS` | `0` | Whole layers preloaded to VRAM, as an integer. Zero, empty or malformed means none; values above 48 are capped at 48. The hybrid recipe uses `qwen4exp_option_b_global_top_k` instead. |
| `ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING` | `1` | Choose the preloaded layers from the end of the network. |
| `ZEROMOE_QWEN4EXP_PREFILL_OP_OFFLOAD` | `1` (default) | Run host-resident experts on the GPU during prefill, by copying the batch's experts across. `0` keeps them on the CPU. |
| `ZEROMOE_QWEN_CPU_THREADS` | all OpenMP threads | Positive integer. Caps (never raises) the threads used for Qwen CPU expert work. |
| `ZEROMOE_QWEN4EXP_HOST_NODE` | off | Use the host-node CPU/GPU schedule (`flag_on`). Measured in [runtime-settings.md](runtime-settings.md). |
| `ZEROMOE_CUDA_SCHEDULE` | `auto` (`blocking` when `ZEROMOE_QWEN4EXP_HOST_NODE=1`) | CUDA synchronisation: `auto`, `blocking` or `spin`. Other values select the default. |

### Qwen memory policy

| Variable | Default | Effect |
| --- | --- | --- |
| `ZEROMOE_QWEN_GRAPH_REGISTER` | on for Qwen | Pin (register) the host expert arenas. Pinned copies run at about 10.6 GB/s against about 4.9 GB/s pageable, which cuts the per-request decode setup from about 3.9 s to 2.0 s. Set `0` to keep the arenas pageable, for machines where the pinned budget is too small. `ZEROMOE_QWEN_GRAPH_NO_REGISTER=1` also disables it. |
| `ZEROMOE_QWEN_SOFT_PIN_GIB` | `6` | Size, in GiB, of the expert tier that a prefill can release. Only that tier is re-uploaded after the prefill, so a second request's setup is about 0.4–0.7 s for 700–1,700-token prompts. `0` disables the tier. The tier is capped at half the split payload. |
| `ZEROMOE_MEM_GUARD_FLOOR_MB` | 20480 | Lower the MemAvailable stop floor (in MB) during long prefills. Used by the 74 GiB pool configurations. |
| `ZEROMOE_DENSE_SCRATCH_RESERVE_MB` | (see source) | Reserve, in MB, for dense-layer scratch memory. Read in `core/attention_layer.cpp`. |

### Qwen early-ready lane (experimental)

The default Qwen recipe (`configs/qwen38-flash.toml`) does not set any of these. They are experimental.

| Variable | Used for |
| --- | --- |
| `ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT`, `_ASYNC_OUTPUT`, `_RESIDENT_TABLES`, `_LAYER13`, `_PREDICT`, `_SKIP_ALL_HOT` | Early-ready expert placement and output copy. Background and measurements: [the graph refactor worklog](docs/option-b-graph-refactor-steps-2026-10-04.md). |
| `ZEROMOE_QWEN_EARLY_READY_STAGE_LAYER`, `_STAGE_VERIFY`, `_STAGE_DELAY_US`, `_STAGE_WORKER_DELAY_US` | Staging ring for the early-ready lane (test hooks). |
| `ZEROMOE_QWEN_EARLY_READY_ROUTE_TRACE` | Log routing decisions of the early-ready lane. Diagnostic. |
| `ZEROMOE_QWEN4EXP_PREFILL_STATE_HANDOFF` | Hand prefill recurrent state to decode without a copy. |

## Expert offload (SSD and host memory)

| Variable | Default | Effect |
| --- | --- | --- |
| `ZEROMOE_OFFLOAD_RETARGET` | on | Retarget offloaded expert reads when SSD offload is active. `0` disables. See [offload-retarget.md](docs/offload-retarget.md). |
| `ZEROMOE_OFFLOAD_EVICT` | (see source) | `none` never returns pages to the OS. Experiment only: memory grows without bound. |

## GLM-5 / Ling / MiMo / DeepSeek families

These are model-specific. The model's own document (`docs/models/<model>.md`) is the authority for its values.

| Variable | Default | Effect |
| --- | --- | --- |
| `ZEROMOE_GLM_DSA_FULL_GRAPH`, `ZEROMOE_LING_FULL_GRAPH`, `ZEROMOE_DEEPSEEK_V41_FULL_GRAPH` | (see source) | Enable the full graph for that family (`flag_on`). |
| `ZEROMOE_GLM_DSA_CPU_LANE` | (see source) | CPU lane for GLM DSA. |
| `ZEROMOE_GLM5NEXT_CPU_THREADS` | (see source) | ggml CPU threads for GLM-5 Next. |
| `ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB` | (see source) | GPU allocation floor for GLM-5 Next. |
| `ZEROMOE_GLM5NEXT_GPU1_HOT_GIB` | (see source) | Hot-expert budget on the second GPU (multi-GPU runs only). |
| `ZEROMOE_GLM5NEXT_GPU1_HOT_PRELOAD` | `1` | Preload the second-GPU hot set. `0` disables. |
| `ZEROMOE_GLM5NEXT_PREFILL_GPU1` | `1` | Use the second GPU during prefill. `0` disables. |
| `ZEROMOE_GLM5NEXT_PREFILL_OP_OFFLOAD` | off | Prefill op offload, as for Qwen (`flag_on`). |
| `ZEROMOE_GLM5NEXT_PILOT_PREFETCH` | `0` | Pilot-driven prefetch depth. `0` disables. |
| `ZEROMOE_GLM5NEXT_PILOT_STAGE` | (see source) | In-graph pilot and VRAM staging ring depth. |
| `ZEROMOE_GLM5NEXT_MODEL_DIRECT` | off | Read model tensors directly (`1` enables). |
| `ZEROMOE_GLM5NEXT_OFFLOAD_SPLIT` | (see source) | Split of expert offload between tiers (same name also as `ZEROMOE_LING_OFFLOAD_SPLIT`). |
| `ZEROMOE_MIMO2_OFFLOAD_DDR_GIB` | (see source) | DDR budget for MiMo expert offload, in GiB. |

## Benchmark and diagnostic

| Variable | Effect |
| --- | --- |
| `ZEROMOE_BENCH_IGNORE_STOP` | Benchmark runs ignore end-of-sequence tokens, so every run generates its full token count. Set to `1` in bench scripts. |
| `ZEROMOE_BENCH_SILENT` | Suppress per-token printing and per-token VRAM sampling in the timed path. |
| `ZEROMOE_STATS_FILE` | Append one JSON line per finished request (same as `--stats-file PATH`): prompt and generated tokens, prefill and decode speed, time to first token, decode setup. Fields: [performance-measurement.md](performance-measurement.md). |
| `ZEROMOE_GENERATED_TOKEN_FILE` | Write each generated token id to this file, one per line (truncated at step 0). For numerical comparisons. |
| `ZEROMOE_DECODE_LOGITS_DUMP` | Dump decode logits to this file. Diagnostic. |
| `ZEROMOE_WUV_INT8` | Attention value-projection int8 path (`1` enables). Diagnostic; read in `core/attention_layer.cpp`. |
| `ZEROMOE_GLM5NEXT_GRAPH_LAYER_TIMING`, `ZEROMOE_MIMO2_NODE_TRACE`, `ZEROMOE_GLM5NEXT_ROUTE_TRACE` | Per-layer or per-node timing and traces. Diagnostic. |

## Not yet documented

Entries marked "(see source)" have a read site but I have not yet confirmed their default or full effect against
the code. They should be completed before this page is used as a reference.
