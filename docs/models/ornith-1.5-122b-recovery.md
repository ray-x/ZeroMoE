# Ornith 1.5 122B: recovered work and remaining gaps

Status: **historical smoke evidence recovered from the Codex transcript; model support is not accepted.** The earlier Ornith-specific source changes and temporary run artifacts are not in the recovered tree. This note preserves what can be reconstructed without treating a short smoke run as parity or performance acceptance.

## Target artifact

- GGUF: `/mnt/usb-ssd/@/home/ray/models/Ornith-1.5-122B-UD-IQ3_XXS.gguf`
- Observed file size: 49,515,667,712 bytes (49.5 GB decimal).
- GGUF architecture: `qwen35moe`.
- Metadata recorded during the earlier inspection: 49 blocks total, with one NextN block and 48 target layers; 3,072 hidden width; 256 routed experts and top-8 routing; 36 recurrent GDN target layers and 12 full-attention target layers. GDN metadata included inner size 8,192, 16 groups, state width 128, and convolution width 4. Full attention used 32 query heads, 2 KV heads, and 256 dimensions per head.
- The IQ3_XXS file contains mixed expert tensor quantizations. The graph binder must use each tensor's actual GGML type.

These figures are transcript-recovered metadata, not a fresh parse or hash of the current model file.

## Work completed before the reset

The first CUDA smoke run placed **zero expert layers in VRAM** and left the host expert arenas pageable. A 65-token prefill took about 100.99 seconds (0.64 tok/s). That run exposed the configuration gap: the graph could execute with settings that were technically valid but far too slow for useful inference.

A later short smoke used the shared Qwen hybrid graph for the `qwen35moe` equations, a complete DDR expert source, and a partial trailing expert block in VRAM. The recorded run used the RTX 3090, context 128, prefill batch 128, `ram_preload_all_experts=true`, and RAM pin settings of 90% / 45 GiB. The allocator reported a 16.85 GiB expert arena with 19 complete resident layers, `[29,48)`, while the other 29 layers remained host-backed. The full expert DDR source was reported at about 38.8 GiB.

For that run:

- 65-token prefill took 3.15 seconds (20.63 tok/s).
- The decode graph produced three output tokens. Its output-interval counter reported 0.288 seconds (6.94 tok/s); the request-level counters also reported 7.46 seconds for the whole request (0.40 tok/s overall) and 4.31 seconds of decode request time (0.70 tok/s). Those timing counters do not reconcile, so 6.94 tok/s should be treated as a short graph-output interval, not a validated steady decode rate.
- The request ended after three generated tokens. No error/NaN was recorded in the summary, but this is only a smoke check.

The old capability log requested 48 trailing device layers and the allocator clamped it to 19. The source was subsequently changed to request 22 by default, but that default was not confirmed by a later Ornith run.

## What is and is not recovered

The exact historical source diff is not recoverable from the current Git refs. The earlier Ornith/Qwen3.5 changes were deliberately reverted when the user chose to restart the model port after the common graph work. The temporary `/tmp/ornith-sanity.toml` and `/tmp/ornith-sanity.log` were also removed during that cleanup. The original standalone commit objects named in the earlier session are absent from this recovered Git object database. The measurements above are therefore transcript evidence only; there is no raw log to re-run the checkers against.

The current `master` has a later `Qwen35MoEGraphModel` adapter for the exact `qwen35moe` architecture and a validated Qwen3.6-35B-A3B IQ2_M full-fit path. That does **not** establish Ornith 1.5 122B support. The existing [Qwen3.6 validation note](qwen36-35b-a3b.md) explicitly says Ornith checkpoints have no captured full-fit gate log, teacher-forced parity run, or performance measurement. The accepted full-fit policy also requires every expert layer on device; the supplied 49.5 GB GGUF cannot fit in the host's combined 40 GiB of VRAM, even before graph state and workspace.

No matched llama.cpp decode run, teacher-forced hidden-state/router comparison, full-fit checker result, or 90%-of-llama.cpp performance result was recovered for this 122B checkpoint. Do not list it as supported from these results.

## Restart checklist

1. Start from the current shared graph adapter, but validate the exact 122B GGUF and its tensor map independently from the Qwen3.6-35B checkpoint.
2. Decide and explicitly plan the intended non-full-fit policy: the historical smoke used a partial VRAM block plus complete DDR expert arenas. Estimate dense weights, resident experts, KV/GDN state, workspace, host RAM, and pinned pages before graph allocation.
3. Save the exact model hash, TOML, command/environment, source and GGML provenance, raw engine log, llama.cpp reference log, and checker outputs under a persistent `build_mtp_results/<date>-ornith-122b-<workload>/` directory. Do not use `/tmp` as the only copy.
4. Compare teacher-forced hidden states and routed expert IDs/weights with llama.cpp, then run the full-graph execution gate. Measure prefill, graph setup, steady decode, and end-to-end throughput separately.
5. Require a matched llama.cpp decode measurement and the project performance gate before marking the 122B model supported.
