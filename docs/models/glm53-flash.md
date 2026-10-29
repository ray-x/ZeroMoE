# GLM 5.3 Flash (GLM5NEXT)

- **Arch / family:** `glm5next` → `GLM5NEXT`. 45 trunk layers (34 KDA + 11 DSA), dense FFN layers
  0-2, MoE layers 3-44 (42) × 288 experts, top-8 + shared expert, `hc_mult=4`.
- **Model:** `/path/to/models/GLM-5.3-Flash/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf` (+ shard 2).
- **Placement (whole-layer):** attention/HC/router/shared/head on the 3090; routed experts of
  layers 3-10 in VRAM, layers 11-44 on CPU (`cpu_affinity_layers = [11..44]`).

## Canonical recipe

```bash
echo "<prompt>" | CUDA_VISIBLE_DEVICES=0 \
  ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=512 ZEROMOE_GLM5NEXT_CPU_THREADS=6 \
  scripts/run_capped.sh -- ./build/zeromoe \
  --config configs/bench/glm53flash_AJ_IQ2XXS_wholelayer8_cpu.toml --temp 0 --predict 150 --force-start
```

The registered GLM5Next graph is the default CUDA path; there is no layered-path environment switch.

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
  refuses, so run under `run_capped.sh` with .
- `ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=512` is required or the dense arena refuses and decode
  silently falls back to legacy.
- A fresh binary's first run spends ~67 s in CUDA JIT (sm_75 PTX on sm_86); warm it once.

## GSQ-RCO 3.0-bit (117 GB, Q2_K/Q3_K/Q4_K experts), offload-pool mode (2026-10-29)

`GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf` does not fit the whole-layer placement (91 GiB of experts), so it runs in offload-pool
mode: experts live in a DDR pool read from the model file, hot ones in VRAM, the CPU lane computes the cold ones.

Recipe (dual GPU, zero SSD misses; box 125 GB DDR, 3090 + 5060 Ti):

```toml
graph_offload_first_layer = 3
graph_offload_model = "/path/to/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf"
graph_offload_ddr_gib = 88      # the cold set is ~88 GiB; 80 leaves ~2 SSD misses/token (190 ms/token)
```

```bash
CUDA_VISIBLE_DEVICES=0,1 ZEROMOE_GLM5NEXT_GPU1_HOT_GIB=14.2 ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=512 \
  scripts/run_capped.sh --limit-gb 118 -- ./build/zeromoe --config <toml from bench/glm_gsq_rco_dual.toml + the keys above> \
  --temp 0 --predict 100 --force-start --graph-offload-usage-file docs/ssd-offload/data/glm53flash_usage_combined_old_plus_survey5.txt
```

| Config (pool 88 GiB, 0 misses) | ms/token | tok/s |
|---|---|---|
| 3090 only, 6 decode threads | 93.8 | 10.7 |
| 3090 + 5060 Ti (14.2 GiB hot), 6 threads | 76.9 | 13.0 |
| same, 10 decode threads (new default) | 70.5-71.1 | **14.1-14.2** |
| same, K-quant GPU assist (1 slot) | 90.7 | 11.0 (worse: leave assist off with two GPUs) |
| 3090 only, pool 80 (2 misses/token, 70 ms of SSD wait) | 190 | 5.3; with K-quant assist 5.8 |

Lane at 14.2 tok/s: wait 21 ms + CPU cold-expert compute 39 ms per token. Measured with the box partly loaded (load 12-14 from
other jobs), so these are slight underestimates.

Changes behind these numbers: decode CPU lane defaults to 10 threads (was 6); the prefill graph's thread count no longer
follows `ZEROMOE_GLM5NEXT_CPU_THREADS` (10 prefill threads grew host memory by ~16 GiB right after pool seeding and tripped the
memory watchdog; prefill stays at 6, override with `ZEROMOE_GLM5NEXT_PREFILL_CPU_THREADS`); in offload mode the engine's host
copies of the dense weights (~8.6 GiB) are freed once the decode graph holds them on the device; the early-ready lane names
the cause (layer, call, input hidden state, weights) before aborting on a non-finite router weight.

Gotchas:
- Pool 88 peaks at ~104 GiB anonymous memory on the 3090-only run (15.7 GiB available, guard 10 GiB) and ~89-92 GiB with the
  5060 Ti holding 14.2 GiB of experts (29 GiB available). Run it only on a quiet box: another job using ~10 GiB trips the
  watchdog. Skipping the dense host copies at startup does NOT work: the layer-by-layer prefill builds its MLA caches from
  host `attn_k_b`/`attn_v_b` and the dense arena size is derived from the host copies (a try gave zeroed weights and a 0.65 s fake prefill).
- Benchmarks are invalid under CPU load: the same run gave 0.89 tok/s while a neighbour compiled (lane compute 323-2123 ms/token).
- Open: at pool 80 (3090 only) some runs hit a non-finite hidden state at decode start, `invalid argument` on dense-weight
  uploads, or runaway memory during seeding. Not reproduced at pool 88; cause not found. Ruled out: lane (messenger only), pool
  pinning, deferred host unregisters, registered upload sources, CUDA API errors (driver log). Notes in the handoff doc.
- 20 tok/s has no measured path yet: per token 39 ms CPU expert compute + 21 ms graph. The next lever is a wider Q2_K CPU dot
  (the patch series has fast dots for IQ2_XXS/IQ1_S/IQ3_XXS/Q2_0 but not Q2_K/Q3_K/Q4_K).
