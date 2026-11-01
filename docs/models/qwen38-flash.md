# Qwen 3.8 Flash (Qwen4Exp)

- **Arch / family:** `qwen4exp` → `Qwen4Exp`. 48 layers, hybrid Gated-DeltaNet + full attention,
  low-rank hyper-connections, PLE n-gram embedding, 512-expert softmax-gated MoE + shared expert.
- **Model:** `/path/to/models/qwen3.8-flash-next/UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf`
- **Placement (split 22/26):** 22 layers GPU-resident, 26 layers CPU (`cpu_affinity_layers`),
  `qwen4exp_cpu_gpu_overlap = true`.

## Canonical recipe (decode baseline)

```bash
ZEROMOE_BIN=./build/zeromoe bench/qwen_decode_baselines.sh split 10 400
```

The script sets `CUDA_VISIBLE_DEVICES=0`, passes
`--graph-device-expert-layers 40 --graph-device-expert-trailing 1`, and uses
`ZEROMOE_BENCH_SILENT=1` / `ZEROMOE_BENCH_IGNORE_STOP=1`. Full-graph prefill/decode are defaults; the
old `ZEROMOE_QWEN4EXP_FULL_GRAPH*` switches are removed. Input
`Tell a story.\n:q\n` (16 tokens). Metric: `[REQUEST-DECODE-WALL]`.

## Frozen baseline

| Metric | Value |
|---|---|
| Decode (band) | **22.0 – 30.0 tok/s** (measured 26.0–26.1; doc 28.66, spread 24.6–28.7) |
| Prefill | 25.7 tok/s (16 tokens) |
| Peak RSS | not captured (single-GPU, ~22 GiB VRAM) |

## Tricks applied (why)

- **Split 22/26** instead of all-GPU: the 24 GiB card cannot hold all 48 layers; CPU overlaps GPU
  via `qwen4exp_cpu_gpu_overlap`.
- **10 threads, not 12:** the CPU part is barrier-bound; a thread sharing a core with another
  process stalls the whole pool (`docs/qwen-decode-baselines-2026-10-16.md` §"Why 10 threads").

## Gotchas

- Read `/proc/loadavg` first: >2 means something else is using cores and the split run is barrier-bound.
- `context_size = 416` (16-token prompt + 400 tokens); a longer gen pushes the last token onto the
  legacy path and fakes a gap.
- If `[MOE-OWNERSHIP]` does not show the 22-layer device block, the graph fell back to legacy decode.

## Q2_0 global-pool recipe: current baseline (v0.0.1, 2026-10-28)

- **Model:** Qwen3.8-Flash-Next GSQ-RCO Q2_0 (two shards, 99 GB), plus the Q4_K_M shared MTP draft checkpoint (1.9 GB).
- **Placement:** all 48 layers keep their attention/dense weights on the 3090; a global ranking pins the 13,000 hottest experts in VRAM (`qwen4exp_option_b_global_top_k`, `graph_device_expert_layers = 0`,
  `graph_device_expert_trailing = false`); the rest run on the CPU. With MTP, 14,000 experts do not leave room for the draft graph.
- **Config:** `configs/bench/qwen38-flash-q2_0-mtp.toml`. Set `model_path` and `spec_draft_model`; for plain decode set `spec_type = "none"`.
- **Run:** `echo "<prompt>" | CUDA_VISIBLE_DEVICES=0 ./build/zeromoe --config configs/bench/qwen38-flash-q2_0-mtp.toml` from the repository root (the pin files are relative paths). Prompts: `bench/prompts/mtp_suite/`.
- **Build:** `scripts/build_ggml.sh <dir>` (applies all of `patches/ggml`, picks AVX-512/VNNI from `/proc/cpuinfo`), then `scripts/build_with_ggml.sh`. Check `python3 tools/build_provenance.py check --binary build/zeromoe --repo .` before quoting a rate.

Frozen baseline (3 passes per cell, medians, tok/s; details and caveats in `configs/bench/perf-baselines.md`):

| prompt | prompt tok | generated tok | plain decode | MTP decode | speed-up | draft acceptance | MTP end-to-end* | prefill (MTP run) |
|---|---|---|---|---|---|---|---|---|
| palindrome (code, low miss) | 29 | 463 | 48.3 (47.7-48.6) | **83.0** (83.0-83.3) | 1.72x | 79.3% | 72.8 | 0.73 s |
| code review | 115 | 500 | 49.3 (49.1-50.3) | **78.3** (77.9-79.3) | 1.59x | 70.9% | 52.0 | 3.18 s |
| B-tree | 47 | 500 | 50.0 (49.8-50.6) | **79.9** (79.6-80.1) | 1.60x | 68.1% | 53.3 | 3.05 s |
| math | 89 | 500 | 47.1 (46.8-47.3) | **80.7** (80.6-81.9) | 1.71x | 84.0% | 53.3 | 3.12 s |
| leetcode | 128 | 246 | 45.0 (44.8-45.8) | **74.3** (73.7-74.3) | 1.65x | 86.8% | 37.4 | 3.20 s |
| story (high entropy) | 43 | 500 | 48.6 (48.4-49.5) | **64.1** (63.4-64.5) | 1.32x | 52.4% | 45.8 | 3.07 s |

Gotchas:

- `qwen4exp_expert_promotion` defaults to `"off"` on master; this baseline needs `"on"`. Without it MTP drops to 77 / 66 / 57 / 50 / 70 / 32.
- The old `mtp_*` TOML keys were renamed: `spec_type = "draft-mtp"`, `spec_draft_model`, `spec_draft_n_max`. The old names are rejected at startup.
- The MTP-specific alternating run favored six graph CPU threads over ten for that batched-window workload. Keep six as an explicit MTP override; ordinary full-graph decode now inherits the affinity-based automatic policy (10 of 12 logical CPUs on this host).
- `ZEROMOE_*` graph switches from the earlier recipes (`..._FULL_GRAPH`, `..._GRAPH_PRELOAD_LAYERS`) are gone; placement comes from the TOML keys above.



## Full-graph layout and decode path

The `qwen4exp` architecture resolves to `Qwen4ExpGraphModel` in `models/graph/full_graph_models.cpp`. The adapter enters the shared Qwen full-graph builder/runtime in `models/qwen/ggml_qwen4exp_graph_*.cpp`; one decode graph contains all 48 layers and is scheduled across CPU and CUDA backends. The graph carries full-attention KV state, GDN recurrent state, hyper-connection state, and PLE state between tokens. With CUDA graph support compiled in, each decode token submits and synchronizes the scheduled graph once, then the next token updates graph inputs and replays it.

The split baseline uses 26 CPU-affinity layers and requests a trailing 40-layer candidate block. CPU affinity takes precedence, leaving 22 complete expert layers in VRAM and 26 layers' experts in host DDR. The logged GPU expert arena is 17.74 GiB. All experts are preloaded into the host source; the GPU arena holds selected resident expert tensors. The separate Q2_0 hotset recipe selects experts globally by the Qwen usage survey (14,554 of 24,576, about 18.74 GiB); unselected experts remain in DDR. These are different placement recipes and their throughput figures must not be mixed.

## Timing evidence and limits

| View | Existing Qwen3.8 evidence | What it does not establish |
|---|---|---|
| Workflow | Split: 16-token prefill about 25.7 tok/s, decode setup about 65 ms, then 400-token decode in 13.96 s (28.66 tok/s). Hotset Q2_0: 43.12 tok/s over 196 decode steps. | The split record has no matched end-to-end request rate; the hotset and split records use different quantization and placement. |
| CPU | With DEBUG plus `ZEROMOE_QWEN4EXP_GRAPH_LAYER_TIMING=1`, the graph callback reports CPU node time per layer/operation and the gap between CPU nodes every ten tokens. A contended 12-thread split recorded 338 ms of CPU node time per token. | That 338 ms sample is not a quiet baseline; the callback excludes CUDA execution and the CPU gap can include waiting for dependent GPU work. |
| GPU | Nsight Systems 2026.1 is available and supports CUDA graph-node tracing on this host. | The archived INFO baseline has no GPU kernel/node timing or GPU per-layer attribution. `nvidia-smi` utilization is not kernel time, and the graph scheduler callback only brackets host submission. |
| Per layer | The current DEBUG logger can identify CPU layer/op work and the first-decode node/backend sequence. | The Qwen3.8 baseline does not contain a complete CPU+GPU per-layer timing map. This must be measured before attributing its speed gap to individual layers. |

Request stats (`prefill_ms`, `decode_setup_ms`, `ttft_ms`, steady decode rate and percentiles, `decode_wall_ms`, and `total_ms`) are the request-level source of truth. Profiled runs should use device-side CUDA graph-node timing for GPU work and the graph callback for CPU work, then keep those artifacts separate from the uninstrumented throughput acceptance run. Do not present the old CPU-only debug counters as a complete GPU/CPU breakdown.


## 2026-10-30 diagnostic profile (not a baseline)

The exact UD-IQ1_M model used for the archived 28.66 tok/s split baseline was not mounted. The available Qwen3.8 IQ3_XXS shards were used for a diagnostic at `/mnt/usb-ssd/@/home/ray/models/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`. Because this quant is larger and the profile had to reduce the trailing candidate block to 37 layers, its throughput must not be compared with the UD-IQ1_M result. The planner selected 19 VRAM layers (15.47 GiB) and 29 CPU/DDR layers, with ten graph CPU threads.

A 20-token, context-128 request recorded prompt=15 tokens, prefill=1,345.9 ms, decode setup=121.0 ms, p50/p95 decode step=587.8/651.5 ms, decode wall=11,415.0 ms (1.75 tok/s), and total=12,819.8 ms. Its graph gate recorded one prefill and 19 decode replays but `memory_plan=unknown`; `check_full_graph_moe_run.py --full-fit` therefore fails. This is execution evidence only, not onboarding acceptance.

Nsight Systems captured CUDA graph nodes for that short request. The device kernel summary totals 268.1 ms across the request; its largest groups are BF16/float matrix-vector kernels, quantized matrix-vector kernels, Q8 quantization, MoE top-k, and GDN. The complete report is [`nsys-kernel-summary.txt`](../../build_mtp_results/2026-10-30-qwen38-flash-3090-attribution/gpu-node-short/nsys-kernel-summary.txt), with the raw 29.8 MB capture in `build_mtp_results/2026-10-30-qwen38-flash-3090-attribution/gpu-node-short/`. This report groups by CUDA kernel name; it does not map GPU work to model layers.

The CPU-only per-layer callback can report operation time by layer and the gaps between CPU nodes. A DEBUG 400-token run emitted those lines, but its raw engine log was overwritten by a later run whose copied TOML retained the parent `log_path`; the extracted samples are not retained as acceptance evidence. A path-correct 20-token DEBUG CPU sample is now preserved, but it has one steady sample at position 24 and the shared-host load rose after startup. It is diagnostic, not an accepted per-layer timing baseline. The CUDA graph gate also remains `Unknown` for this Qwen4Exp path, so the Qwen3.8 historical performance rows are useful profile references but do not satisfy the current full-fit memory-plan gate.


### Path-correct CPU layer sample (2026-10-30)

The `cpu-layer-corrected/` run records one sample at position 24: CPU node work 28.283 ms, CPU-node gap 36.851 ms, and 29 CPU-affinity layers with individual CPU totals from 0.62 to 1.66 ms. The largest displayed layer was L20 at 1.66 ms; L28 and L30 were 1.34 and 1.29 ms. Per-operation CPU sums are in the same engine log. The run used the IQ3_XXS shards, context 128, 15 prompt tokens, 20 generated tokens, and candidate block 37. A later host load sample reached 19.55, so this is not a clean performance sample. The full-fit gate still rejects the run because `memory_plan=unknown`.

The separate Nsight profile provides GPU kernel totals by kernel symbol but no per-layer mapping. A complete per-layer CPU+GPU timing map remains unavailable; model-level throughput and the full-fit gate remain unresolved for this IQ3_XXS diagnostic.
