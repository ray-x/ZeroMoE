# Configuring ZeroMoE

Start with the matching model recipe in `configs/`. Change the paths, memory
budgets, GPU topology and context size for your machine; keep its measured
backend choices unless you are benchmarking. No "fast/balanced" preset
silently changes numerical or memory policy.

```bash
./build-cuda/zeromoe --config configs/qwen38-flash.toml --prompt "Hello"
```

## Choose the right surface

| Surface | Purpose |
| --- | --- |
| Model TOML recipe | Persistent model paths, topology, budgets, context and generation settings |
| Normal CLI / `--help` | Common per-run overrides |
| `--help-all` | Compatibility flags, advanced tuning, tracing and benchmarks |
| Root `graph_*` TOML keys / `--help-settings` | Advanced typed graph tuning; reference includes effects, units, family defaults and legacy names |
| Legacy `ZEROMOE_*` environment variables | Existing scripts and controlled experiments, not required setup for the Qwen recipe |
| CMake presets/options | Platform/backend capabilities compiled into the binary, not per-request tuning |

`--help-all` and `--help-settings` belong to the full CUDA engine (`zeromoe`).
The separate `zeromoe_cpu` executable has a smaller CLI and uses `CPUConfig`,
including `omp_max_threads`. It rejects `graph_*` keys rather than pretending
to apply CUDA graph tuning. Use a CPU configuration, not a CUDA model recipe.

## Advanced graph tuning and precedence

The implemented graph resolver uses:

**common defaults < family defaults < root TOML settings < legacy environment**.

There is no new generic CLI override or profile-pin layer for these fields yet.
Existing non-graph CLI flags keep their existing behavior.

Place graph settings **before the first `[[gpus]]` table**. Do not add a
`[graph]` section: the lightweight TOML readers do not support nested settings.
For example, Qwen's old launch-time placement flags become:

```toml
# Root settings, before [[gpus]]
graph_device_expert_layers = 0
graph_device_expert_trailing = true
# Optional: cap Qwen CPU expert workers; 0 keeps available-thread selection.
# graph_cpu_threads_decode = 10
```

Qwen full-graph prefill/decode already default to enabled. The shipped
`configs/qwen38-flash.toml` no longer requires its four historical launch
flags. It retains its existing measured backend, hotset and memory settings.
Whole-layer placement and partial hotset budgets remain separate concepts;
trailing placement has no whole-layer effect while the layer count is zero.

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
effective value/source (`default`, `family`, `toml:graph_...`, or `env:ZEROMOE_...`).
Family overrides include their reason. A legacy alias logs a migration warning.
If both TOML and a legacy alias are set, **the environment still wins**.
Remove the exported variable after migrating, or your TOML change will not take
effect. Historical environment parsing is preserved, including its inconsistent
boolean and malformed-number behavior; it is not the stricter canonical parser.

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
backend. They do not change inference tuning. Hardware-specific build choices
still matter: for example, an x86 CPU without AVX-512 VNNI needs
`-DZEROMOE_ENABLE_AVX512_VNNI=OFF` when configuring. See
[the build instructions](../README.md#build-and-installation-pipeline).
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
