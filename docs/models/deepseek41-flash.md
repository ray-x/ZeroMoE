# DeepSeek V4.1 Flash (DeepSeek_V41)

- **Arch / family:** `deepseek41` -> `DeepSeek_V41`. 40 layers, 384 experts, top-6, ratio-1/2
  compressed MLA, hyper-connections, conditional-memory Engram layers.
- **Status: performance baseline only, NOT accepted as supported.** Four rounding-changing
  optimizations below have not been compared with llama.cpp (teacher-forced), and the 90%-of-llama.cpp
  and `check_full_graph_moe_perf.py` gates have not been run. See "Open acceptance gaps".

## Current baseline (2026-11-08, v102)

Hardware: RTX 3090 24 GB (PCIe Gen3 x16) + RTX 5060 Ti 16 GB (Gen3 x8), Xeon W-2235 (6C/12T), 131.6 GB DDR.
Model: `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-0000{1..5}-of-00006.gguf` (IQ2_XXS mix with IQ1_M/IQ2_XS
experts) + `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` sidecar, on the local NVMe (`direct_io = true`).
Workload: `bench/deepseek41/workload_959tok_prompt.txt` (959 prompt tokens, 64 generated, temp 0, ctx 4096).
Source: `d7372d9`, clean tree; no route log or tensor tracing enabled; `ram_pinned_budget_gb = 110` (whole DDR slab page-registered); GGML `8baabd53`; CUDA 13.3.73; Release.

| Request | Prefill tok/s | Decode tok/s (steady, first-to-last output) |
|---|---:|---:|
| 1 | 88.50 | 12.93 |
| 2 | 90.75 | 13.15 |

Decode p50 about 75 ms/token; start to ready about 76 s (was about 91 ms at v74). `tools/check_full_graph_moe_run.py --full-fit --require-full-graph-prefill` passes.
Raw evidence and history: `docs/handoff/deepseek-v41-flash-current.md` (v70-v102).

## How to run

Memory: the startup gate wants about 127.9 GB free on a 131.6 GB host (`host_ram_gb = 118.6`; strict full
residency needs >= ~118). Stop other large processes first; no swap is allowed in the guarded run.

```bash
# 1. Build (Release, provenance-stamped; re-run cmake after any source edit or the stamp is rejected)
cmake -S . -B build_dsv41 -DCMAKE_BUILD_TYPE=Release && cmake --build build_dsv41 -j
# 2. Serve (config uses model paths under /mnt/usb-ssd/DeepSeek-V4.1-Flash-1017-GSQ-RCO/; edit if different.
#    A local NVMe copy of the six GGUFs cuts start-to-/health from ~324 s to ~62 s; throughput within 2-3%)
CUDA_VISIBLE_DEVICES=0,1 NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=10 \
  ./build_dsv41/zeromoe --serve --config configs/bench/deepseek_v41_iq2xxs_fullres_dual_gpu.toml \
  --host 127.0.0.1 --port 18081 --alias deepseek-v41-flash
# 3. Send the workload (chat completion, temp 0, max_tokens 64) with the prompt in
#    bench/deepseek41/workload_959tok_prompt.txt as the user message; repeat 3 times.
# 4. Read [REQUEST-DECODE-OUTPUT] (decode) and "Prefill (whole pass)" / Rates (prefill) from the log, then
python3 tools/check_full_graph_moe_run.py <server.log> --full-fit --require-full-graph-prefill
```

Startup takes minutes (preloads ~12.8k experts into DDR, pins the rest in VRAM). Prefer the guarded harness
(`systemd-run` scope with `MemorySwapMax=0`, memory gate, host-pressure watchdog `tools/host_pressure_watchdog.py`).
Key config values: `full_moe_vram_dynamic_slots = 160`, `graph_engram_cache_mb = 128`, `attn_score_threads = 8`,
`full_moe_host_headroom_gb = 5.0`, `host_ram_startup_margin_gib = 4.0`, `prefill_pin_file = ""`, and the
decode pin file `configs/bench/deepseek_v41_iq2xxs_decode_pins.txt` (observed decode routes of this workload;
regenerate for other workloads). Rejected settings: `OMP_WAIT_POLICY=active`, global `OMP_NUM_THREADS=6`,
96 dynamic slots, `attn_score_threads` 6/12.

## What changed to get 3.7 -> 10.8 decode and 31/66 -> 71 prefill

1. **Grouped single-expert CPU kernel** (`core/cpu_expert_executor.cpp`): decode misses with one expert use the
   grouped IQ2 path (+11%). Added an IQ2_XS grouped dot (`backends/cpu/kernels_cpu.cpp`).
2. **Q5_K/Q4_K/Q6_K `attn_output_a` on GPU** (`grouped_output_proj_dense_host`): decode 4.2 -> 6.7.
3. **Parallel Engram row reads**: OpenMP `pread` of distinct missing rows, for prefill (cold 31 -> 66) and for
   single-token decode (removed ~12 ms/token of serial USB reads).
4. **Per-phase CPU threads**: decode runs half the hardware threads, prefill the full set.
5. **GPU per-head attention for decode** (`kernels/models/dsv41_head_attention.cu`, per-layer device mirrors of the
   window/compressed KV, inverse RoPE on host): decode -> ~10.8.
6. **Memory plan**: 160 shared dynamic VRAM slots + `host_ram_gb = 118.6` (prefill >= 70); VRAM pool fix so startup
   pins do not consume the shared dynamic slots (`core/gpu_topology.hpp`, fixes layer-0 dummy preflight failure).

## Open acceptance gaps (do not mark supported)

- Items 1, 2 and 5 and the IQ2_XS dot change rounding; each flips 2.5-7% of decode route lines against the
  previous build. No teacher-forced comparison against llama.cpp (`scripts/validate_vs_llama.sh`) has been run.
- No matched llama.cpp performance run (90% bar) and no `check_full_graph_moe_perf.py --minimum-tps`.
- One machine, one workload, three requests; memory margin is tight (~127.9 GB gate on 131.6 GB).

## Older recipe (superseded; cache-backed graph, 2026-10-21)

### Old recipe

```bash
echo "Explain what DNS is and how a domain name gets resolved to an IP address." \
  | CUDA_VISIBLE_DEVICES=0,1 \
    scripts/run_capped.sh -- ./build/zeromoe \
    --config configs/bench/deepseek_v41_100gib_ddr_17gib_vram_dual_gpu.toml --temp 0 --predict 64 --force-start
```

Metric: `[REQUEST-DECODE-OUTPUT]` (or `[REQUEST-DECODE-WALL]`).

### Old baseline (historic, 1.8-3.2 tok/s decode)

| Metric | Value |
|---|---|
| Decode (band) | **1.8 – 3.2 tok/s** (measured 2.18–2.22 cold; doc 2.90 cold / 4.0–4.5 steady) |
| Prefill (20 tokens) | 0.9 tok/s (startup-dominated) |
| Peak RSS | ~92 GiB |

## Tricks applied (why)

- **Cache-backed graph** (one host callback per layer) reuses the native RAM/VRAM cache; graph
  owns **zero** MoE weight bytes (`docs/deepseek_v41_changelog_2026-10-21.md`).
- Whole-prompt prefill remains the default. Use `prefill_batch_tokens` or llama.cpp-compatible
  `-ub/--ubatch-size` only when an explicit graph-batch limit is needed.
- `pin_ram_fraction=1.0` + `ram_pinned_budget_gb=40` + `pin_global_tiers` (default off) cut SSD
  reads and lift decode 5-11%.

## Numerical trace

Use `--trace-dir DIR` to capture float32 boundary tensors as `.f32` files. Optional
`--trace-layer N` and `--trace-pos N` select one layer and position. The
profiling runner exposes the same selectors and passes them to the binary as
CLI options. These are developer diagnostics; no environment aliases remain.

## Gotchas

- Decode is SSD-read + H2D-upload bound (cold misses ~55-70 ms + upload ~75 ms/token). The "steady
  state" number is higher than a cold run; the band is set for the cold 64-token recipe.
- The cold-run number varies with prompt (different routing → different cold-miss rate); to confirm
  a regression, re-run with the exact reference prompt.
- Benign teardown crash (`CUDA error: driver shutting down`) after metrics print — pre-existing.
