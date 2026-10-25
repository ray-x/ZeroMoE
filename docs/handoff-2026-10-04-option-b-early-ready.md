# Handoff: Option-B early-ready graph scheduling

**Date:** 2026-10-04
**Workspace:** `/home/ray/Projects/ZeroMoE`
**State:** all-split prototype implemented and validated; uncommitted; paused at the user's request.

## Result

The all-split early-ready scheduler path is opt-in through `ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT=1`. It starts the GPU hot/shared work, copies each split layer's activation and route IDs into private buffers, then runs a prepared cold-expert GGML graph on the main thread after an input-ready CUDA event. Private buffers prevent the outer GGML allocator from reusing memory still in flight. The normal scheduler barriers and arithmetic paths remain in use when the flag is unset. The Layer-13-only proof remains available as `ZEROMOE_QWEN_EARLY_READY_LAYER13=1`.

On the Q2_0 checkpoint and RTX 3090, with the right-sized 0/44/4 layout and 10 CPU threads, six guarded, order-balanced pairs measured:

| Mode | Mean decode rate |
|---|---:|
| Baseline | 31.754 tok/s |
| Early-ready, all 44 split layers | 32.478 tok/s |
| Difference | +2.28%, 6/6 paired wins |

The paired gains ranged from +1.13% to +3.34%. These numbers apply to this model, prompt/history and host. The BF16 cuBLAS experiment was disabled. Interrupted runs caused by other inference or compiler jobs were retained in the discarded-run directory and excluded; the benchmark checks host activity during each run.

The baseline and all-split candidate also produced byte-identical logits for 64 records, with zero KL, total variation and maximum logit delta. Their 64 free-running token IDs matched, and the generated explanation was inspected. `gguf_tool verify` passed for the checkpoint. The full native test suite passed after both the Layer-13 and all-split changes.

## Files changed for this work

- `include/qwen4exp_early_ready.hpp`: private pinned buffers, prepared cold CPU graph, event handling, per-replay guards and the scheduler callback.
- `src/ggml_qwen4exp_graph.cpp`: marker insertion, per-layer context ownership, explicit GPU-branch expansion and the two opt-in flags.
- `bench/option_b_early_ready_check.py`: fixed-history, free-running and guarded six-pair runs; records interference and preserves invalid attempts.
- `patches/ggml-scheduler-early-ready-layer13.patch`: isolated patch for the integration GGML scheduler hook. It is applied in `../llama.cpp-moestream-integration` and must stay paired with the engine hook.
- `docs/option-b-scheduler-overlap-design-2026-10-04.md`: architecture, measurements and detailed reproduction notes.

## Evidence and reproduction artifacts

- Throughput summary and audit: `logs/option_b_early_ready_all_perf_20261004/{summary.json,results.json,manifest.json}`.
- Raw-logit comparison: `logs/option_b_early_ready_all_logits_20261004/comparison.json`.
- Free-running output and generated IDs: `logs/option_b_early_ready_all_free_20261004/`.
- Native test log: `logs/option_b_early_ready_all_native_test_20261004.log`.
- Checkpoint verification: `logs/option_b_early_ready_model_verify_20261004.log`.
- Performance retry/admission history: `logs/option_b_early_ready_all_perf_resume_20261004.log`; invalid runs are preserved under `logs/option_b_early_ready_all_perf_20261004/discarded/`.
- Six-pair command:

```bash
python3 bench/option_b_early_ready_check.py \
  --engine ../moestream-build-upstream-master/moestream \
  --out-dir logs/option_b_early_ready_all_perf_20261004 \
  --prompt-ids logs/quant_decode_compare_chat_20261004/prompt.tokens \
  --continuation logs/quant_decode_compare_chat_20261004/Q2.tokens \
  --reps 6 --all-split --wait-idle 300
```

## Next decision

Review the six-pair result against the project's 1.5% acceptance bar, then decide whether this remains an experimental opt-in or should be integrated into a runtime configuration. It is currently off by default. No files were committed. Do not clean the working tree: it contains many unrelated user changes and generated artifacts from earlier work. BF16 routing remains deferred, and `GGML_CUDA_BF16_MMVF_CUBLAS=0` was used for these measurements.
