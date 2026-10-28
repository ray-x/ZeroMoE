# Ling 3.1 Flash (BailingMoE3)

- **Arch / family:** `bailingmoe3` → `BailingMoE3`. KDA + gated MLA, 72 layers, MoE layers 4+
  (68) routed experts.
- **Model:** `/path/to/models/ling-3.1-flash/Ling-3.1-flash-AD-IQ1_S-00001-of-00004.gguf`
- **Placement:** no-copy offload graph; 72 GiB DDR pool, 3.0 + 12.5 GiB GPU caches, 3700 pinned
  VRAM slots, GPU assist 3 + assist2 1.

## Canonical recipe (short decode, 32 tokens)

```bash
LING_DDR_GIB=72 LING_CAP_GIB=95 LING_STOP_GIB=20 \
LING_GPU0_CACHE_GIB=3.0 LING_GPU1_CACHE_GIB=12.5 \
LING_PINNED_VRAM_CAPACITY=3700 LING_OFFLOAD_PINNED=0.5 \
LING_GPU_ASSIST=3 LING_ASSIST2_SLOTS=1 LING_CPU_THREADS=10 \
LING_USAGE_FILE="$PWD/bench/ling/reference-audit/usage-hi-there-robust4.pin" \
LING_LOG=/tmp/ling-short.log LING_BIN=./build/zeromoe \
bench/ling/reference-audit/run.sh
```

Metric: `[REQUEST-DECODE-OUTPUT]` (first output to last output), *not* the shutdown `Rates:` line
(`docs/ling-3.1-flash-benchmark-method.md`).

## Frozen baseline

| Metric | Value |
|---|---|
| Decode (band) | **9.0 – 11.5 tok/s** (measured 10.2–10.3; doc 10.15) |
| Prefill (2 tokens) | ~1.5 tok/s (startup-dominated) |

## Tricks applied (why)

- **No-copy offload graph**: the graph aliases existing VRAM slots and streams cold experts from
  the DDR pool; no MoE weights are copied into the graph.
- **`ZEROMOE_GLM5NEXT_BATCHED_NEGATIVE_SKIP=1`** for prompts >64 tokens (patched negative-id
  skip) — without it the valid-dummy fallback aliases tokens onto expert 0 and over-runs its row
  range (`CUDA error: illegal memory access` at layer 4).

## Gotchas

- The decode target (10 tok/s) is met only at the threshold with no margin; treat a drop below 9.0
  as a regression.
- `LING_BIN` must point at the candidate binary — the runner otherwise hardcodes `build-ling/zeromoe`.
- Requires the host-node hook (early-CPU lane) or decode falls to ~3 tok/s.
