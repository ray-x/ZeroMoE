# MiMo baseline check — provisional

Run date: 2026-10-28 (Australia/Sydney). ZeroMoE source e63b96329acc58144262a0fade8eae72c90b810c, built clean in worktree codex/mimo-env-cleanup. The model was read in place from /mnt/usb-ssd/@/home/ray/models/MiMo-V2.6-Flash-RL-GSQ-RCO-3bit.gguf (115,678,863,104 bytes); no model copy was made.

## ZeroMoE run

- Profile: configs/bench/mimo26_flash_gsq_rco_3bit_ingraph_dual.toml, with the real model path.
- Maintained survey overrides: --graph-offload-ddr-gib 100 and --graph-offload-usage-file bench/profiles/mimo26_combined.usage.
- Workload: four held-out prompts with prompt token counts 25, 26, 17, and 24; 64 generated tokens per prompt; six CPU threads; profile context capacity 2048; RTX 3090 + RTX 5060 Ti.
- Environment: CUDA_VISIBLE_DEVICES=0,1, ZEROMOE_DISABLE_ROUTER_BF16=1, ZEROMOE_BENCH_IGNORE_STOP=1, ZEROMOE_BENCH_SILENT=1. The default in-graph MoE path was used.
- Memory: systemd cap 110 GiB, swap disabled; peak ZeroMoE RSS 78.92 GB. Model/profile startup and population took about five minutes.
- Prefill: 92 prompt tokens in 2.856 seconds, aggregate 32.21 tok/s. Per-prompt prefill rates: 26.14, 33.89, 34.82, 37.22 tok/s.
- Steady graph decode: 252 graph runs in 16.161 seconds, 15.59 tok/s. Per-request decode-output rates: 16.29, 13.89, 15.89, 15.20 tok/s. This sample clears the documented 10 tok/s target.
- Graph gate: policy=required, memory_plan=unknown, four graph prefill runs, 252 graph decode runs, zero layered S=1 forwards, zero legacy S=1 expert dispatches.

## llama.cpp reference

- Clean source and GGML base: 8baabd53b77d1c48c6ef8d3b620436292f3f6231.
- Build: llama.cpp 0.7.0-dev build 11785, GNU 16.2.1, CUDA 13.3.73, native targets 86;120a; NCCL unavailable.
- Placement: --n-cpu-moe 47, --n-gpu-layers 99, CUDA0/CUDA1, layer split, 60/40 tensor split, six threads, batch/ubatch 512. Expert tensors stayed CPU-side; this llama-bench target does not expose --moe-cache-mib.
- Three-repeat combined pp/tg rows: pp25/tg64 11.22 tok/s (sd 0.05); pp26/tg64 11.15 (sd 0.44); pp17/tg64 6.11 (sd 4.24); pp24/tg64 3.76 (sd 1.74). The pp17 and pp24 rows varied substantially.

## Acceptance status

Treat this as a provisional baseline. ZeroMoE's profile reserves context 2048; llama-bench sets n_ctx to prompt+generation (81–90 tokens). The reference uses CPU expert tensors, while ZeroMoE uses pinned VRAM/DDR placement. The same GGUF, active prompt lengths, generation lengths, GPUs, and thread count were used, but those context and placement differences prevent a final relative-performance claim.

python3 tools/check_full_graph_moe_run.py ... --full-fit fails because the Mimo graph summary reports memory_plan=unknown. The full-fit performance checker is also not an acceptance gate for this run: it expects one prompt length and a llama-bench row with no CPU MoE placement. Mimo remains in progress pending routed-expert/hidden-state parity, explicit memory-plan evidence, and repeatable matched-context measurements.

Artifacts: exact TOML and prompts, command/environment records, ZeroMoE build manifest, stdout and engine log, llama-bench JSON/stderr, and the full-graph check result are in this directory. build_mtp_results/2026-10-28-mimo-baseline-e63b963/ contains the earlier exploratory invocation that omitted the two graph-offload overrides and ended at 36/64 generated tokens; do not treat it as the baseline.
