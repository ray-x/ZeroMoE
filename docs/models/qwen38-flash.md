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
| palindrome (code, low miss) | 29 | 463 | 48.6 (48.4-49.2) | **82.3** (81.6-82.4) | 1.70x | 79.3% | 71.7 | 0.74 s |
| code review | 115 | 500 | 49.1 (48.7-49.2) | **78.5** (75.5-78.8) | 1.60x | 70.9% | 52.0 | 3.18 s |
| B-tree | 47 | 500 | 50.2 (49.9-50.3) | **79.1** (78.0-79.8) | 1.57x | 68.1% | 52.9 | 3.07 s |
| math | 89 | 500 | 46.6 (45.7-47.0) | **80.8** (80.3-81.4) | 1.73x | 84.0% | 53.1 | 3.14 s |
| leetcode | 128 | 246 | 44.7 (44.4-45.7) | **72.5** (71.3-72.6) | 1.62x | 86.8% | 36.9 | 3.21 s |
| story (high entropy) | 43 | 500 | 48.9 (48.8-50.3) | **64.4** (64.1-64.5) | 1.32x | 52.4% | 46.0 | 3.07 s |

Gotchas:

- `qwen4exp_expert_promotion` defaults to `"off"` on master; this baseline needs `"on"`. Without it MTP drops to 77 / 66 / 57 / 50 / 70 / 32.
- The old `mtp_*` TOML keys were renamed: `spec_type = "draft-mtp"`, `spec_draft_model`, `spec_draft_n_max`. The old names are rejected at startup.
- The graph CPU team defaults to the physical core count (6 here). A seventh thread lands on an SMT sibling and costs about 8%.
- `ZEROMOE_*` graph switches from the earlier recipes (`..._FULL_GRAPH`, `..._GRAPH_PRELOAD_LAYERS`) are gone; placement comes from the TOML keys above.

