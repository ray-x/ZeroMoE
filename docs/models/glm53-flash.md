# GLM 5.3 Flash (GLM5NEXT)

- **Arch / family:** `glm5next` → `GLM5NEXT`. 45 trunk layers (34 KDA + 11 DSA), dense FFN layers
  0-2, MoE layers 3-44 (42) × 288 experts, top-8 + shared expert, `hc_mult=4`.
- **Model:** `/home/ray/vllm-workspace/GLM-5.3-Flash/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf` (+ shard 2).
- **Placement (whole-layer):** attention/HC/router/shared/head on the 3090; routed experts of
  layers 3-10 in VRAM, layers 11-44 on CPU (`cpu_affinity_layers = [11..44]`).

## Canonical recipe

```bash
echo "<prompt>" | CUDA_VISIBLE_DEVICES=0 ZEROMOE_GLM5NEXT_FULL_GRAPH=1 \
  ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=512 ZEROMOE_GLM5NEXT_CPU_THREADS=6 \
  scripts/run_capped.sh -- ./build_master/zeromoe \
  --config configs/bench/glm53flash_AJ_IQ2XXS_wholelayer8_cpu.toml --temp 0 --ngen 150 --force-start
```

`ZEROMOE_GLM5NEXT_FULL_GRAPH=1` is REQUIRED — without it the legacy per-layer path runs ~6× slower.

## Frozen baseline

| Metric | Value |
|---|---|
| Decode (band) | **6.5 – 8.5 tok/s** (measured 7.5–7.6; doc 7.26) |
| Prefill | 0.5 tok/s (29 tokens; dominated by startup) |

## Tricks applied (why)

- **Fused hyper-connection ops** are the default (graph ~6.1k nodes vs ~31k unfused) — this is the
  single biggest decode win on this file (`docs/GLM5.3-Flash-baseline.md` §3.4).
- **LM head inside the graph** removes a ~48 ms/token CPU round-trip.

## Gotchas

- `host_ram_gb=106` needs `106 + 21.5 = 127.5 GiB` of `MemAvailable`; on a 125 GiB box preflight
  refuses, so run under `run_capped.sh` with `--force-start`.
- `ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=512` is required or the dense arena refuses and decode
  silently falls back to legacy.
- A fresh binary's first run spends ~67 s in CUDA JIT (sm_75 PTX on sm_86); warm it once.
