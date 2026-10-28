# DeepSeek V4.1 Flash (DeepSeek_V41)

- **Arch / family:** `deepseek41` → `DeepSeek_V41`. 40 layers, 384 experts, top-6, ratio-1/2
  compressed MLA, hyper-connections, conditional-memory Engram layers.
- **Model:** `/home/ray/vllm-workspace/DeepSeek-V41-Flash` (7 shards, Q2_K).
- **Placement:** dual GPU — 3090 (17 GiB main expert pool) + 5060 Ti (13.5 GiB), ~100 GiB DDR,
  active pools 48/16 slots.

## Canonical recipe

```bash
echo "Explain what DNS is and how a domain name gets resolved to an IP address." \
  | CUDA_VISIBLE_DEVICES=0,1 \
    scripts/run_capped.sh -- ./build/zeromoe \
    --config configs/bench/deepseek_v41_100gib_ddr_17gib_vram_dual_gpu.toml --temp 0 --predict 64 --force-start
```

Metric: `[REQUEST-DECODE-OUTPUT]` (or `[REQUEST-DECODE-WALL]`).

## Frozen baseline

| Metric | Value |
|---|---|
| Decode (band) | **1.8 – 3.2 tok/s** (measured 2.18–2.22 cold; doc 2.90 cold / 4.0–4.5 steady) |
| Prefill (20 tokens) | 0.9 tok/s (startup-dominated) |
| Peak RSS | ~92 GiB |

## Tricks applied (why)

- **Cache-backed graph** (one host callback per layer) reuses the native RAM/VRAM cache; graph
  owns **zero** MoE weight bytes (`docs/deepseek_v41_changelog_2026-10-21.md`).
- `pin_ram_fraction=1.0` + `ram_pinned_budget_gb=40` + `pin_global_tiers` (default off) cut SSD
  reads and lift decode 5-11%.

## Gotchas

- Decode is SSD-read + H2D-upload bound (cold misses ~55-70 ms + upload ~75 ms/token). The "steady
  state" number is higher than a cold run; the band is set for the cold 64-token recipe.
- The cold-run number varies with prompt (different routing → different cold-miss rate); to confirm
  a regression, re-run with the exact reference prompt.
- Benign teardown crash (`CUDA error: driver shutting down`) after metrics print — pre-existing.
