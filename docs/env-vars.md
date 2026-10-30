# Environment variables

Prefer the model recipe and canonical root `graph_*` TOML keys for migrated
graph settings. See [configuration and precedence](configuration.md), or run
`zeromoe --help-settings` for effects, actual family defaults and aliases.
Legacy aliases still work and override TOML, but now log migration warnings.
Many experimental controls remain environment-only; this is a staged migration. CPU thread count is shared across model families and uses `-t/--threads`.

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
| `ZEROMOE_QWEN4EXP_PREFILL_CHUNK` | Removed 2026-10-30 | Use llama.cpp-compatible `-ub/--ubatch-size` or TOML `prefill_batch_tokens`; the default remains whole-prompt prefill. |
| `ZEROMOE_QWEN4EXP_PREFILL_OP_OFFLOAD` | Removed 2026-10-30 | Use shared TOML `graph_prefill_op_offload` or `--graph-prefill-op-offload`; the Qwen family default remains on. |
| `ZEROMOE_QWEN4EXP_PREFILL_DEVICE_STATE` | unset | With TOML `qwen4exp_prefill_state_handoff = true`, set to `0` to keep zero-state inputs on host and upload them for prefill. This lower-VRAM fallback is disabled for prompt-cache/MTP and incompatible staging paths. |
| `ZEROMOE_GGML_GPU_ROUTER` | Removed 2026-10-30 | Use TOML `moe_ggml_gpu_router`; this Qwen4Exp-only switch selects GPU router/top-k, while `false` keeps the host router. |
| `ZEROMOE_GDN_PROJ_GGML` | Removed 2026-10-30 | Use TOML `gdn_proj_ggml = true` to enable fused GDN projections. The per-call CPU/GGML fallback remains. |
| `ZEROMOE_GDN_FULL_GGML` | Removed 2026-10-30 | Use TOML `gdn_full_ggml = true` for the full GDN GGML graph; it requires `gdn_proj_ggml = true`. |
| `ZEROMOE_HC_GGML` | Removed 2026-10-30 | Use TOML `hc_ggml = true` to select the Qwen4Exp GGML HC path; the CPU implementation remains the default and fallback. |
| `ZEROMOE_QWEN_CPU_THREADS` | Removed 2026-10-29 | Use the shared `-t/--threads` option for CPU work across models. |
| `ZEROMOE_QWEN4EXP_HOST_NODE` | off | Use the host-node CPU/GPU schedule (`flag_on`). Measured in [runtime-settings.md](runtime-settings.md). |
| `ZEROMOE_QWEN_HOST_NODE_THREADS` | Removed 2026-10-29 | Host-node CPU workers now share the global `-t/--threads` limit. |
| `ZEROMOE_CUDA_SCHEDULE` | `auto` (`blocking` when `ZEROMOE_QWEN4EXP_HOST_NODE=1`) | CUDA synchronisation: `auto`, `blocking` or `spin`. Other values select the default. |
| `ZEROMOE_QWEN4EXP_EXPERT_PROMOTION` | Removed 2026-10-30 | Use TOML `expert_promotion` (`off`/`observe`/`on`); graph-native MTP enables promotion automatically. |
| `ZEROMOE_PROMO_CAP`, `ZEROMOE_PROMO_BURST`, `ZEROMOE_PROMO_MIN_MISSES`, `ZEROMOE_PROMO_FREQ_WEIGHT`, `ZEROMOE_PROMO_DECAY_EVERY` | Removed 2026-10-30 | Use TOML `graph_promo_cap`, `graph_promo_burst`, `graph_promo_min_misses`, `graph_promo_freq_weight`, `graph_promo_decay_every` or the matching `--graph-promo-*` flags. Defaults 12, 20, 2, 4096, 16000000. |
| `ZEROMOE_MTP_PREFILL_OVERLAP` | Removed 2026-10-30 | Use TOML `graph_mtp_prefill_overlap` or `--graph-mtp-prefill-overlap`; default off. |
| `ZEROMOE_MTP_GREEDY_FAST_PATH` | Removed 2026-10-30 | The direct argmax fast path is always used at temperature 0. |
| `ZEROMOE_QWEN4EXP_ROUTE_IDS_DUMP` | Removed 2026-10-30 | Route-ID capture was removed; existing dumps remain usable offline. |

### Qwen memory policy

| Variable | Default | Effect |
| --- | --- | --- |
| `ZEROMOE_QWEN_GRAPH_REGISTER` | on for Qwen | Pin (register) the host expert arenas. Pinned copies run at about 10.6 GB/s against about 4.9 GB/s pageable, which cuts the per-request decode setup from about 3.9 s to 2.0 s. Set `0` to keep the arenas pageable, for machines where the pinned budget is too small. `ZEROMOE_QWEN_GRAPH_NO_REGISTER=1` also disables it. |
| `ZEROMOE_QWEN4EXP_PHASE2_ALLOW_RISKY_ALLOC` | Removed 2026-10-30 | Phase-2 decode now always enforces the free-VRAM estimate plus 768 MiB margin before compact-arena allocation. |
| `ZEROMOE_QWEN_SOFT_PIN_GIB` | Removed 2026-10-30 | Use TOML `graph_soft_pin_gib` or `--graph-soft-pin-gib`; default 6 GiB. Size of the expert tier that a prefill can release; `0` disables it. Currently Qwen full-graph decode, intended to generalize. |
| `ZEROMOE_MEM_GUARD_FLOOR_MB` | 10240 | Experiment-only override for the MemAvailable stop floor (in MB); values below 8192 are ignored. The shipped long-prompt GLM recipe sets 16384, which raises the current 10 GiB default. |
| `ZEROMOE_DENSE_SCRATCH_RESERVE_MB` | `256` | Reserve, in MB, for dense-layer scratch memory before the VRAM budget checks run. Non-positive or malformed values fall back to 256. Read in `core/attention_layer.cpp`. |

### Qwen early-ready lane (experimental)

The default Qwen recipe (`configs/qwen38-flash.toml`) does not set any of these. They are experimental.

| Variable | Used for |
| --- | --- |
| `ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT`, `ZEROMOE_QWEN_EARLY_READY_ASYNC_OUTPUT`, `ZEROMOE_QWEN_EARLY_READY_RESIDENT_TABLES`, `ZEROMOE_QWEN_EARLY_READY_LAYER13`, `ZEROMOE_QWEN_EARLY_READY_PREDICT`, `ZEROMOE_QWEN_EARLY_READY_SKIP_ALL_HOT` | Early-ready expert placement and output copy. Background and measurements: [the graph refactor worklog](option-b-graph-refactor-steps-2026-10-04.md). |
| `ZEROMOE_QWEN_EARLY_READY_PREDICT_LAYER` | Restrict the pilot prediction to one target layer. Unset predicts for every split layer. |
| `ZEROMOE_QWEN_EARLY_READY_RIDGE` | Path to a pilot ridge table. A missing or invalid file leaves the plain pilot. |
| `ZEROMOE_QWEN_EARLY_READY_STAGE_LAYER`, `ZEROMOE_QWEN_EARLY_READY_STAGE_VERIFY` | Select stage layer (`all` or an integer) and enable output verification for the active staging checks. |
| Retired: `ZEROMOE_QWEN_EARLY_READY_ROUTE_TRACE` | Route records now use `[EARLY-READY-ROUTE]` DEBUG log lines; `bench/option_b_topic_check.py` extracts them into route JSONL artifacts. |
| `ZEROMOE_QWEN4EXP_PREFILL_STATE_HANDOFF` | Removed 2026-10-30 | Use TOML `qwen4exp_prefill_state_handoff`; the A/B script writes the key in a temporary config for each run. |

## Expert offload (SSD and host memory)

| Variable | Default | Effect |
| --- | --- | --- |
| `ZEROMOE_OFFLOAD_RETARGET` | on | Retarget offloaded expert reads when SSD offload is active. `0` disables. See [offload-retarget.md](offload-retarget.md). |
| `ZEROMOE_OFFLOAD_EVICT` | `MADV_DONTNEED` | Set to `free` to evict with `MADV_FREE` instead: pages stay mapped until the kernel reclaims them, so the next read skips the page fault, at the price of a soft memory budget. Any other value (or unset) uses `MADV_DONTNEED`. |
| `ZEROMOE_FULL_GRAPH_USAGE_FILE` | family default | Usage-frequency profile used to rank full-graph resident experts during graph setup. Distinct from `graph_offload_usage_file`, which selects the offload hot set. |

## GLM-5 / Ling / MiMo / DeepSeek families

These are model-specific. The model's own document (`docs/models/<model>.md`) is the authority for its values.

| Variable | Default | Effect |
| --- | --- | --- |
| ~~`ZEROMOE_GLM_DSA_FULL_GRAPH`~~, ~~`ZEROMOE_LING_FULL_GRAPH`~~, ~~`ZEROMOE_DEEPSEEK_V41_FULL_GRAPH`~~ | removed | Full graph is the default for these families and cannot be disabled through the environment. Ling's native/hybrid graph selectors were retired 2026-10-29; `ZEROMOE_DEEPSEEK_V41_FULL_GRAPH_DECODE=0` remains an explicit native-decode comparison. |
| `ZEROMOE_GLM_DSA_CPU_LANE` | off | Advertise the GLM DSA CPU lane: cold experts may run on a private CPU lane concurrently with the layer's GPU work (the early-ready/gpu-assist capability). Only a leading `1` enables it; without it, a mixed VRAM+DDR placement logs a `PLACEMENT-WARN`. |
| `ZEROMOE_GLM5NEXT_EARLY_STATE_EXPORT` | Removed 2026-10-29 | Early state/history expansion is fixed on; measurements showed lower prefill workspace and faster long prompts with matching output. |
| `ZEROMOE_GLM5NEXT_CPU_THREADS`, `ZEROMOE_GLM5NEXT_PREFILL_CPU_THREADS`, `ZEROMOE_DEEPSEEK_V41_CPU_THREADS`, `ZEROMOE_MIMO2_CPU_THREADS` | Removed 2026-10-29 | CPU thread counts now use the shared `-t/--threads` option across models. |
| `ZEROMOE_FULL_GRAPH_READ_THREADS` | Removed 2026-10-29 | Boot source loading uses a fixed two-reader policy to bound temporary memory. |
| `ZEROMOE_GLM5NEXT_GRAPH_PRELOAD_LAYERS` | Removed 2026-10-29 | Use shared `graph_device_expert_layers` TOML key or `--graph-device-expert-layers` for GLM/Ling whole-layer placement. |
| `ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB` | `2048` | Free-VRAM floor the graph allocator must leave before it builds GLM-5 Next graphs, in MiB. `0` removes the floor; it also gates pilot staging. |
| Retired: `ZEROMOE_GLM5NEXT_GRAPH_STRICT` | Removed 2026-10-30 | The shared graph planner already fails a declined graph when the plan fits; `bench/glm53/run_dsa_graph2gpu.sh` checks full-fit graph counters with `check_full_graph_moe_run.py`. |
| `ZEROMOE_GLM5NEXT_CPU_LM_HEAD` | unset | ZeroMoE-specific low-VRAM fallback: keep `output_norm.weight` and `output.weight` on CPU instead of staging about 1.2 GiB on GPU. The CPU path was measured about 48 ms/token slower; the graph GPU head remains the default. |
| `ZEROMOE_GLM5NEXT_GPU1_HOT_ON_GPU0` | Removed 2026-10-29 | The one-off placement verification path was retired; hot-expert buffers always use GPU1. |
| `ZEROMOE_GLM5NEXT_GPU1_HOT_USAGE` | Removed 2026-10-29 | Use shared `graph_offload_usage_file` TOML key or `--graph-offload-usage-file` for both placement and offload profiles. |
| `ZEROMOE_GLM5NEXT_GPU1_HOT_GIB` | unset | Hot-expert budget on the second GPU, in GiB (multi-GPU runs only). Unset or empty leaves the GPU1 hot set unconfigured. |
| `ZEROMOE_GLM5NEXT_GPU1_HOT_PRELOAD` | Removed 2026-10-29 | GPU1 hot-set preload is always attempted at startup; failure still warns and retries upload on first decode build. |
| `ZEROMOE_GLM5NEXT_PREFILL_GPU1` | `1` | Use the second GPU during prefill. `0` disables. |
| `ZEROMOE_GLM5NEXT_PREFILL_CHUNK` | Removed 2026-10-30 | Use llama.cpp-compatible `-ub/--ubatch-size` or TOML `prefill_batch_tokens`; the GLM default remains 2048 tokens. |
| `ZEROMOE_GLM5NEXT_PREFILL_ROUTE_CAPTURE` | Removed 2026-10-30 | Prefill routes are collected automatically when TOML `prefill_pin_file` is set and `freeze_usage_stats = false`; the separate prefill usage profile remains supported. |
| `graph_*` settings | Typed TOML keys and generic `--graph-*` CLI flags. | Offload and scheduler settings use canonical names such as `--graph-prefill-op-offload 1`, `--graph-pilot-copy 0`, and `--graph-offload-split-reads 1`; legacy GLM5Next env aliases are retired. |
| `ZEROMOE_GLM5NEXT_PILOT_PREFETCH` | `0` | Pilot prefetch depth, clamped to 0–8; above `0` enables the pilot. The effective depth is at least `ZEROMOE_GLM5NEXT_PILOT_STAGE`, and predicted experts that are not in host memory are read from SSD ahead of the layer that needs them. |
| `ZEROMOE_GLM5NEXT_PILOT_STAGE` | `0` | Number of next-layer cold experts the pilot stages into the VRAM ring per layer, clamped to 0–8. `0` disables pilot staging. |
| `ZEROMOE_GLM5NEXT_MODEL_DIRECT` | off | Read model tensors directly (`1` enables). |

## Benchmark, diagnostic and test

| Variable | Effect |
| --- | --- |
| `ZEROMOE_BENCH_IGNORE_STOP` | Removed 2026-10-29 | Use llama.cpp-compatible `--ignore-eos` to generate through end-of-generation tokens. |
| `ZEROMOE_PREFILL_UNBATCHED` | Removed 2026-10-30 | Use llama.cpp-compatible `-ub/--ubatch-size 1` or TOML `prefill_batch_tokens = 1` for the S=1 prompt parity path. |
| `ZEROMOE_SCORE_PREFILL_LEGACY` | Removed 2026-10-30 | Prefill uses the GPU score path when configured; the existing CPU path remains the fallback if GPU scoring or its KV mirror is unavailable. |
| `ZEROMOE_GDN_CHUNKED` | Removed 2026-10-30 | The unbenchmarked runtime chunked A/B path was retired; GDN uses its sequential recurrent path. The standalone chunk-form CPU helper remains covered by the mathematical reference test. |
| `ZEROMOE_BENCH_SILENT` | Suppress per-token printing and per-token VRAM sampling in the timed path. |
| `ZEROMOE_QWEN4EXP_MTP_WINDOW=0` | Force the graph-native Qwen MTP target to verify drafted tokens stepwise instead of in a multi-token window. ZeroMoE-specific correctness comparison; default uses the window derived from `spec_draft_n_max`. |
| `ZEROMOE_GENERATED_TOKEN_FILE` | Write each generated token id to this file, one per line (truncated at step 0). For numerical comparisons. |
| `ZEROMOE_DECODE_LOGITS_DUMP` | Dump decode logits to this file. Diagnostic. |
| `ZEROMOE_PROBE_DUMP` | Tool-local offline predictor study: writes per-layer post-attention residuals and router scores for `bench/next_layer_probe_fit.py`; it is not a runtime predictor setting. |
| `ZEROMOE_DUMP_ACTIVATION_LAYER`, `ZEROMOE_DUMP_ACTIVATION_EXPERT`, `ZEROMOE_DUMP_ACTIVATION_OUT` | Tool-local real-activation capture for the optional CPU quantization check. Selects one layer/expert and writes its input as raw float32; the output path defaults to `/tmp/real_activation.bin`. |
| `ZEROMOE_ROUTE_LOG` | Tool-local text capture of final routed expert IDs per layer and token; consumed by `bench/dsv41_pilot_replay.py`. |
| `ZEROMOE_ROUTER_IN_LOG` | Tool-local binary router-input capture for offline DeepSeek V4.1 pilot replay; consumed by `bench/dsv41_pilot_replay.py`. |
| `ZEROMOE_QWEN4EXP_DIAG_DUMP_DIR` | With `--verbosity 5`, write layer-0 Qwen CPU graph tensor snapshots into this directory. |
| `--verbosity 5` | Emits Qwen layer-0 CPU graph diagnostics (`[QWEN4EXP-LAYER0-DIAG]`); the separate dump-directory variable controls binary snapshots. |
| `ZEROMOE_WUV_INT8` | Attention value-projection int8 path (`1` enables). Diagnostic; read in `core/attention_layer.cpp`. |
| `ZEROMOE_TEST_EXL3_CPU_ONLY`, `ZEROMOE_TEST_EXL3_ONLY` | Stop the native test harness early: after the CPU/EXL3 checks with `_CPU_ONLY`, after the CUDA kernel checks with `_ONLY` (the form `README.md` uses for the short CUDA check). |
| `ZEROMOE_TEST_GLM53_DIR` | Removed 2026-10-30 | Use `zeromoe_tests --glm53-dir <path>` (or `--glm53-dir=<path>`). |
| `ZEROMOE_QWEN4EXP_PROBE_LAYER` | Tool-local | With `ZEROMOE_QWEN4EXP_PROBE_DUMP` set, selects the layer whose detailed gate/up/activation/down taps are retained; documented in the full-graph onboarding guide. |
