# Configuring ZeroMoE

Start with [`configs/starter.toml`](../configs/starter.toml) for a first run: it
sets only `model_path` and a small context, leaving VRAM and host-RAM sizing to
the engine. Once a model runs, move to the matching model recipe in `configs/`
and change the paths, memory budgets, GPU topology and context size for your
machine; keep its measured backend choices unless you are benchmarking. No
"fast/balanced" preset silently changes numerical or memory policy.

```bash
./build/zeromoe --config configs/starter.toml --prompt "Hello"
```

## Choose the right surface

| Surface | Purpose |
| --- | --- |
| Model TOML recipe | Persistent model paths, topology, budgets, context and generation settings |
| Normal CLI / `--help` | Common per-run overrides |
| `--help-all` | Compatibility flags, advanced tuning, tracing and benchmarks |
| Root `graph_*` TOML keys / generated `--graph-*` flags / `--help-settings` | Advanced typed graph tuning; reference includes effects, units, family defaults and legacy names |
| Legacy `ZEROMOE_*` environment variables | Existing scripts and controlled experiments, not required setup for the Qwen recipe |
| CMake presets/options | Platform/backend capabilities compiled into the binary, not per-request tuning |

`--help-all` and `--help-settings` belong to the full CUDA engine (`zeromoe`).
The separate `zeromoe_cpu` executable has a smaller CLI and uses `CPUConfig`,
including `omp_max_threads`. It rejects `graph_*` keys rather than pretending
to apply CUDA graph tuning. Use a CPU configuration, not a CUDA model recipe.

## llama.cpp-compatible option names

The common CLI settings use the names from the pinned llama.cpp CLI/server
reference (source commit `8baabd53b77d`; `tools/cli/README.md`,
`tools/server/README.md`, `docs/speculative.md`, and `docs/multi-gpu.md)):
names and control meaning are aligned; model-recipe defaults remain
ZeroMoE/model-specific.

| Intent | llama.cpp | ZeroMoE |
| --- | --- | --- |
| Model, prompt | `--model`, `--prompt` | same |
| Context, output length | `-c/--ctx-size`, `-n/--predict` | same |
| CPU threads | `-t/--threads` | same; caps ZeroMoE's process-wide OpenMP team |
| Sampling | `--temp`, `--top-p`, `--repeat-penalty`; TOML `temperature`, `top_p`, `repeat_penalty` | same llama.cpp option names |
| System prompt | `--system-prompt` | same |
| Single main GPU | `--main-gpu` | same for single-device recipes |
| Log file and verbosity | `--log-file`, `--verbosity` | same |
| Server alias and key | `--alias`, `--api-key` | same; the key environment variable remains `ZEROMOE_API_KEY` |
| Speculative MTP | `--spec-type`, `--spec-draft-model`, `--spec-draft-n-max` | same names; ZeroMoE currently supports `none` and `draft-mtp` |

For multi-GPU ZeroMoE runs, use the recipe's `[[gpus]]` tables. Those tables
also assign dense/MoE roles and per-device cache budgets; they do not have a
one-to-one equivalent to llama.cpp's `--device`, `--split-mode`, and
`--tensor-split`. ZeroMoE's MTP `mtp_graph` key selects its graph-native
implementation and remains ZeroMoE-specific.

## ZeroMoE-specific settings retained for review

These controls have no exact llama.cpp counterpart but configure core
ZeroMoE capabilities. They remain documented for review rather than being
treated as ordinary tuning switches:

| Capability | Canonical controls | Why it is distinct |
| --- | --- | --- |
| Expert residency and offload | `graph_device_expert_layers`, `moe_cache_gb`, `graph_offload_*` | Separate DDR source, pinned DMA arena, VRAM expert cache, and graph placement |
| Usage and pin profiles | `pin_file`, `prefill_pin_file`, `graph_offload_usage_file`, `graph_usage_record_file` | Create/consume route-frequency profiles to choose resident experts |
| Per-device MoE topology | `[[gpus]]` roles, cache sizes, reservations, and active-layer pools | Assigns trunk, hotset, and expert-cache responsibilities per GPU |
| Expert-storage I/O | `direct_io`, `io_uring`, `drop_pages` | Selects ZeroMoE's expert-slab I/O path; no direct llama.cpp CLI equivalent |
| Host resource safety | `host_ram_gb`, memory guard and pinned-memory budgets | Bounds ZeroMoE's host expert tier and startup admission |
| Phase reuse | `graph_phase_cache_mib`, `graph_phase_cache_reserve_mib` | Optional decode-hot expert allocation with explicit phase lifetime |

Unlisted experiment flags should move into developer-only tools or be removed
after their result is finalized. They should not grow the normal run interface.

## Advanced graph tuning and precedence

The graph resolver uses:

**common defaults < family defaults < root TOML settings < legacy environment < CLI**.

Every registered graph setting also has a generated `--graph-<setting-name>` CLI
flag. Convert underscores to hyphens; for example, `graph_prefill_chunk` can be
overridden with `--graph-prefill-chunk 256`. CLI values use the same strict
canonical parser as TOML. Existing non-graph CLI flags keep their existing
behavior.

Place graph settings **before the first `[[gpus]]` table**. Do not add a
`[graph]` section: the lightweight TOML readers do not support nested settings.
For example, Qwen's old launch-time placement flags become:

```toml
# Root settings, before [[gpus]]
graph_device_expert_layers = 2      # 0 = no whole-layer block (global-pool recipe)
graph_device_expert_trailing = false
# Optional: cap Qwen CPU expert workers; 0 keeps available-thread selection.
# graph_cpu_threads_decode = 10
```

Qwen full-graph prefill/decode already default to enabled. The shipped
`configs/qwen38-flash.toml` no longer requires its four historical launch
flags. It retains its existing measured backend, hotset and memory settings.
Whole-layer placement and partial hotset budgets remain separate concepts.
`graph_device_expert_layers` is 2 by default; `0` is the explicit sentinel for
"no whole-layer block" (the global-pool recipe), and an enabled block must be
at least 2 layers. `graph_device_expert_trailing` defaults to false (leading
block) and only affects placement when the layer count is nonzero.

Canonical keys use real TOML types: `true`/`false` for booleans, integers for
counts, numeric fractions in `[0,1]`, and strings for paths or `"auto"`.
Only settings documented as automatic accept `"auto"`. Unknown names,
invalid types/ranges, keys inside GPU tables, and settings unsupported by
the detected model adapter fail explicitly.
GLM offload/lane keys also require Ling's GLM-backed or hybrid implementation;
the native-only Ling variant rejects those unused settings. The full engine
rejects graph tuning in CPU-only mode or without its ggml CUDA backend.

Examples for other families, not Qwen:

```toml
# MiMo: prompt chunk length, not context size.
graph_prefill_chunk = 512
graph_prefill_op_offload = "auto"
# GLM/Ling: DMA pinning can follow GPU-assist availability.
# graph_offload_pin_dma = "auto"
```

Only set values justified for your model and hardware. This is not a generic
recommended tuning block.

At initialization, `[SETTINGS]` prints each applicable canonical key and its
effective value/source (`default`, `family`, `toml:graph_...`, `env:ZEROMOE_...`,
or `cli:--graph-...`).
Family overrides include their reason. A legacy alias logs a migration warning.
If both TOML and a legacy alias are set, the environment wins unless the CLI
sets that same key. Remove the exported variable after migrating so TOML remains
effective. Historical environment parsing is preserved, including its
inconsistent boolean and malformed-number behavior; TOML and CLI use the
stricter canonical parser.

MiMo2 and Ling-prefixed graph-setting aliases have been removed; their maintained
launchers use canonical settings and `--graph-*` flags. Remaining Qwen and
GLM aliases are transitional.

Graph settings are bound before initialization and then immutable. Changing
environment variables or rewriting the config while the process runs is not a
supported way to retune a graph.

## Builds without a wall of `-D` flags

With CMake 3.25 or newer:

```bash
cmake --preset cpu
cmake --build --preset cpu --parallel 4

# Linux + NVIDIA CUDA:
cmake --preset cuda
cmake --build --preset cuda --parallel 4
```

Both presets select Release and ggml; CUDA additionally builds ggml's CUDA
backend. They do not change inference tuning. Configure detects the host CPU:
an x86 host without the AVX-512 baseline is rejected with the missing flags,
and `GGML_AVX512_VNNI` is enabled only when the host has VNNI. Override with
`-DGGML_AVX512_VNNI=ON|OFF`, or `-DZEROMOE_CHECK_HOST_CPU=OFF` when building
for a different machine. See
[the build instructions](../README.md#build-and-installation).
Manual CMake and existing Make targets remain supported.

## Scope and maintenance

This migration adds canonical TOML support for fields already in the shared
MiMo/GLM/Ling resolver plus Qwen whole-layer placement and its shared CPU thread
cap. It does not remove every environment variable, replace experimental
backend selectors, implement execution profiles, or turn compile-time platform
guards into runtime settings. Existing recipes and aliases remain compatible.
Model equations, graph-failure behavior and memory planning are unchanged.

Inspect the remaining surface without guessing undocumented defaults:

```bash
make inventory-settings
make inventory-settings ARGS=--json
make test-settings BUILD_DIR=build-cpu
# Alternatively, after configuring the CPU preset:
cmake --build --preset settings-tests
ctest --preset settings-tests
```

The inventory is lexical and includes aliases, diagnostics, TOML readers,
registered graph fields, CLI names, build options and capability macros.
Its totals are **not** a count of independent user decisions.
`python3 tools/config_inventory.py --check` guards migrated Qwen readers against
bypassing the resolver. Resolver, default-snapshot and startup-binding tests
need neither CUDA nor model weights. Linux/CUDA and teacher-forced model runs
are still required before claiming numerical or performance parity.
