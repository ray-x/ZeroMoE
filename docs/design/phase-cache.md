# Shared prefill/decode VRAM reuse

**Paused, 2026-10-23:** MiMo passed the complete decode check; Ling failed numerical parity even with GPU assist
disabled. The feature remains opt-in. See [the handoff](../handoff-2026-10-23-phase-cache.md) before resuming.

The permanent expert cache remains at fixed addresses. `PhaseExpertCache` owns a second CUDA allocation for
decode-hot experts; it is released before prefill and rebuilt afterwards from pinned, ready DDR experts.
MiMo and the GLM/Ling in-graph path use the same allocation, copy, route-partition and lifetime implementation.

Enable explicitly (MiB, not decimal GB):

```bash
scripts/run_capped.sh --limit-gb 110 -- build_mimo/zeromoe --config <model.toml> \
  --graph-phase-cache-mib 1024 --graph-phase-cache-reserve-mib 512
```

The default extension is zero, so existing profiles retain their fixed placement. The requested extension is
capped by measured free VRAM minus the decode reserve. Keep the existing permanent-cache prefill reserve: the
extension consumes it only during decode. Dense weights, KV and recurrent state do not move into this allocation.
The current implementation extends the core GPU; it does not yet plan independent extensions on several GPUs.

## Ordering and ownership

1. Before prefill, retire decode graphs and their early-ready workers, then drain/free the extension.
2. Build/run the prefill graph using the permanent hot set and existing cold path.
3. Before decode, retire prefill workspace while preserving model/engine-owned state.
4. Select ranked cold experts whose DDR data is both ready and pinned. Copy their gate/up/down planes directly
   from the offload arena into the separate allocation. No demand loading, model-file reads or DDR eviction
   occurs in this operation. Immutable selected identities survive subsequent phase transitions.
5. Build a new decode graph with an extra GPU branch. Remove those identities from the cold mask/remap;
   preserve authoritative route IDs, route slots and router weights. The permanent and optional hot branches
   are disjoint and every contribution is combined exactly once.

Slot strides use the engine's per-layer row alignment, including mixed quantization. The extension owns its
weight descriptors and GPU route maps. It does not create another DDR cache or claim permanent topology slots.

`PlacementEpoch` publishes a new generation on allocation and retirement. A graph holds a lease on its active
generation; release refuses while any graph borrower remains. Graph reuse checks token shape, model identity
and placement generation, and GLM's run paths reject stale bindings before executing. MiMo re-prepares before
execution. Old captured graphs are retired with their schedulers instead of replayed against recycled addresses.

If allocation headroom or pinned DDR sources are unavailable, the extension stays disabled and execution uses
the permanent cache. No SSD read is issued to make an extension fit. This guarantee concerns extension refills:
a model whose normal DDR+VRAM placement is incomplete may still have its existing demand misses.

## Validation

```bash
cmake --build build_mimo -j4 --target zeromoe graph_settings_test layer_plan_test phase_route_test
build_mimo/graph_settings_test tests/graph_settings_defaults.snapshot
build_mimo/layer_plan_test
build_mimo/phase_route_test
python3 bench/phase_cache_transition.py --model mimo --output logs/phase-mimo
python3 bench/phase_cache_transition.py --model ling --isolate-assist --output logs/phase-ling
```

The CUDA test runs short → chunked long → repeated short requests, captures first-step logits before sending
the next request, and compares every decode step against the fixed-cache control under per-request teacher
forcing (`ZEROMOE_TEACHER_FORCE_REQUEST_DIR`). It checks repeated-prompt state reset,
monotonic placement generations, stable extension capacity, completed release/refill cycles, finite logits
and measured GPU memory. Logs record refill/release times. MiMo requires zero decode store loads; Ling records
its baseline demand loads separately because its standard reference profile cannot hold all expert identities.

`--control-only` collects an independent reference; `--control-output <case-directory>` reuses saved reference
logits only after verifying the model/workload configuration. The third request uses the first short request's
oracle. It still executes the real repeated request in the candidate, so state-reset checks are not skipped.
These are targeted correctness/memory checks, not a rerun of the frozen performance bands.

The first validation exposed a MiMo-specific private upload of `token_embd.weight`, which undid CPU embedding
placement and consumed 0.66 GB of prefill workspace. MiMo now omits this unused tensor from graph-private
weight accounting and uploads; the engine still honors an explicit GPU embedding setting.
