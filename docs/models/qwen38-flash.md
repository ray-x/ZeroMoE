# Qwen 3.8 Flash (Qwen4Exp)

- **Arch / family:** `qwen4exp` → `Qwen4Exp`. 48 layers, hybrid Gated-DeltaNet + full attention,
  low-rank hyper-connections, PLE n-gram embedding, 512-expert softmax-gated MoE + shared expert.
- **Model:** `/data/sdb/models/qwen3.8-flash-next/UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf`
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
