# GLM 5.3 (GLM_DSA)

- **Arch / family:** `glm-dsa` → `GLM_DSA`. 78 layers, 75 routed-expert layers, absorbed MLA + DSA,
  routed MoE.
- **Model:** `/path/to/models/GLM-5.3-GGUF-align` (6 shards, UD-IQ2_M).
- **Offload store:** `/data/glm53_offload.gguf` (routed experts, first layer 3); experts also read
  from the model files (split reads across both drives).
- **Placement:** dual GPU — 5060 Ti core (dense/attention/KV/LM head) + 3090 expert vault (22 GiB
  hot set), 64 GiB DDR pool, SSD-offloaded cold experts.

## Canonical recipe

```bash
systemd-run --user --scope -p MemorySwapMax=0 \
  env ZEROMOE_BIN=/tmp/msbin \
  bench/glm53/run_ab.sh baseline \
  ZEROMOE_GLM5NEXT_GPU1_HOT_GIB=22 OFFLOAD_DDR_GIB=64 NGEN=80 \
  ZEROMOE_GLM5NEXT_PILOT_PREFETCH=3 ZEROMOE_GLM5NEXT_PILOT_STAGE=3 \
  ZEROMOE_GLM5NEXT_MODEL_DIRECT=1
```

(`/tmp/msbin` is a neutral-named symlink to the binary — `bench/glm53/preflight.sh` greps command
lines for the literal `zeromoe` and would false-positive on a spelled-out path.)

## Frozen baseline

| Metric | Value |
|---|---|
| Decode (band) | **1.8 – 2.4 tok/s** (measured 2.14–2.17; doc 2.00–2.09) |
| Prefill (22 tokens) | 0.8 tok/s (startup-dominated) |
| Peak RSS | ~90 GiB (64 GiB pool run) |

## Tricks applied (why)

- **Page-reusing eviction (`ZEROMOE_OFFLOAD_RETARGET`, default on)**: `mremap(MREMAP_DONTUNMAP)`
  hands an evicted expert's populated pages to the incoming expert instead of
  `madvise(DONTNEED)` + refault — the single biggest win on this file (+17-19 %,
  `docs/offload-retarget.md`).
- **CPU lane** (`ZEROMOE_GLM_DSA_CPU_LANE=1`) runs cold experts concurrently with GPU work; needs
  the host-node hook.
- **Graph LM head** removes a ~50 ms/token CPU round-trip.

## Gotchas

- Requires the host-node hook; without it the lane is silently disabled and decode drops ~3×.
- `ZEROMOE_GLM5NEXT_GPU1_HOT_GIB=22` is the ceiling (23 GiB does not fit).
- Benign teardown crash (`CUDA error: driver shutting down`) after metrics print — pre-existing.
