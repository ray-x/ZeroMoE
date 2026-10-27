# Handoff: Qwen4Exp coherence and staged decode performance (2026-10-07)

## Objective and current status

The active task is to fix the Qwen4Exp staged/Phase-2 decode coherence issue, then measure and explain its decode slowdown against the accepted early-ready path. Coherence is the first gate. The accepted production path remains early-ready private CPU graph + immutable resident route tables + asynchronous cold-output upload.

No fix has been committed in this task. The arena-ownership fix is already commit `521d67a`; the prefix-handoff follow-up and staged-path work in this checkout are uncommitted. Do not treat the task as complete.

## Correct checkout and base

Use the bug-fix checkout:

```text
/tmp/zeromoe_bugfix_521d67a_20261007
branch: bugfix/staged-coherence-521d67a
HEAD: 521d67a773b3b9e22f48bb5fcad0f6840ca663a6
```

The root `/home/ray/Projects/ZeroMoE` is on `cleanup/ggml-exl3-only`, which is the wrong work branch for this investigation. The two checkouts share base commit `521d67a`, but their dirty source trees differ. A six-pair comparison run during this session accidentally used `/home/ray/Projects/ZeroMoE/build_ggmlcuda/zeromoe`; discard that result as evidence for this task.

The intended checkout already has `build_ggmlcuda/zeromoe` (the r7 correctness manifest records SHA256 `5bb109b567ca7c155f1a83c644ebea26deb59c7d21af28cbe6ffdd4434366e80`). Its CMake source directory is `/tmp/zeromoe_bugfix_521d67a_20261007`, but its runpath points to prebuilt GGML libraries under the wrong root checkout. The local `ggml-source` copy matched `/home/ray/Projects/beellama.cpp/ggml` in the comparison made on 2026-10-07; still inspect and pin the GGML source/library path in the new clean folder before trusting a rebuilt binary. If rebuilding is needed, use `-j10` and verify `CMakeCache.txt`, `ldd`, and the binary hash afterward. The target branch has no new CUDA source changes in the relevant diff.

Skip EXL3 changes. The checkout has unrelated EXL3 files and other user modifications; do not include them in this bug-fix scope or commit.

## Known fixes and remaining coherence question

- Commit `521d67a` fixes staged expert arena ownership. A staged child that borrows from a graph which itself borrows the retained decode arena now follows the source-graph chain to the canonical owner for its cursor and address map. The prior bug let a staged tensor overlap an Option-B resident weight and overwrite it.
- The uncommitted `refresh_prefix_handoff()` follow-up treats the retained graph-owned CPY output as authoritative instead of refreshing it from a stale scheduler intermediate. A five-token teacher-forced check passed 5/5 top-1 against early-ready (mean KL `0.000581`). The 64-token comparison still differs at step 6.
- Other historical Qwen4Exp fixes already in the base lineage include the GDN device-state seed fix, the KV-cache scatter-index transpose fix, and the RoPE mode/frequency fix. The memory records are `gdn-full-device-state-seed-bug.md`, `qwen4exp-decode-kv-cache-transposed-index-bug.md`, and `qwen4exp-rope-imrope-mismatch-root-cause.md`. Reopen them only if the new trace points to a regression in those paths.

The correct branch's retained r7 comparison is `logs/phase2_bugfix_hidden_pair_20261007_r7/`:

- Early-ready versus the staged candidate: 62/64 top-1 agreement, mean KL `0.000674`, disagreements at steps 6 and 27. Step 6 has a small reference margin (`0.0131`); the reference top-1 is candidate rank 2.
- The first hidden-state difference between those two ZeroMoE arms is position 0, layer 36 (RMS `0.000183`, max absolute `0.00440`). The saved full per-layer comparison is `hidden_comparison.json`.
- The original 95.3% figure was early-ready ZeroMoE versus staged ZeroMoE, not versus llama.cpp.

## Standard llama.cpp check and its caveat

The reference used here is the CPU path from `/home/ray/Projects/llama.cpp`, HEAD `3cf03257f219afbe7334045ff7c6a06ac68c627d`, build 0.4.1-dev. It loads the same split Q2_0 GGUF and receives explicit prompt/continuation token IDs. The output harnesses and logs are retained in `/tmp/llama_standard_20261007/`.

On the same forced token history, the bug-fix branch’s early-ready baseline agrees with incremental llama.cpp on 58/64 top-1 predictions (90.625%); mean KL is `0.105572`. First disagreement is step 3: llama top-1 `9522`, ZeroMoE `4145`, reference margin `0.07019`, KL `0.99987`. Disagreement steps are `[3, 7, 9, 26, 27, 36]`. The staged r7 arm agrees on 57/64 (89.0625%), with the same six disagreements plus step 6. Thus the baseline-to-llama difference is already present before staging; staging adds the step-6 top-1 flip.

The decode-only aligned l_last comparison begins at position 28/layer 0 (RMS `0.000317`, max `0.00424`); its delta grows gradually through depth (at position 28, layer 47 RMS `0.182`) and does not first jump at layer 13. A later comparison of the retained 28-token ZeroMoE prefill dump against llama.cpp full-batch per-position states found an earlier difference: position 0/layer 0 (RMS `0.000218`, max `0.01459`), growing gradually to mean RMS `0.0302` by layer 47 across the prompt. Therefore the cross-engine difference is already present during prefill; it is not introduced by incremental decode or Layer-13 staging. The target branch’s early-ready logits match the older retained known-good `run23` baseline on all 64 top-1s (mean KL `0.000892`), so this is not a new regression from the staged fix.

The user recalled a possible llama.cpp decode-path issue. Project memory does not document a llama inference bug; its documented historical state bugs are in ZeroMoE. A direct internal llama control did find path-dependent numeric differences:

- One `llama_decode` over the whole forced sequence with `n_ubatch=128` versus prompt-then-single-token incremental decode: 63/64 top-1s agree; first flip at step 3 (`9522` vs `13983`), KL `0.00287`.
- One `llama_decode` over the whole sequence with `n_ubatch=1` versus incremental: 63/64 agree, but the flip moves to step 51 (`760` vs `22238`), KL `0.00195`.
- The full-batch versus incremental l_last trace first differs at position 27/layer 3 (RMS `3.12e-5`, max `2.28e-4`) and reaches about `0.025` RMS at layer 47, position 30.

This establishes a llama batch-shape-sensitive near-top-1 flip, not a verified stale-state/corruption bug. It accounts for one disagreement location, not the other five baseline-to-llama disagreements. Keep incremental llama.cpp as the user-requested coherence reference, while treating individual flips near this numeric limit cautiously and continue tracing the five common disagreements.

Key retained reference artifacts:

- `/tmp/llama_standard_20261007/bugfix_branch_baseline_vs_llama.json`
- `/tmp/llama_standard_20261007/bugfix_branch_staged_vs_llama.json`
- `/tmp/llama_standard_20261007/bugfix_branch_baseline_hidden_vs_llama.json`
- `/tmp/llama_standard_20261007/bugfix_branch_staged_hidden_vs_llama.json`
- `/tmp/llama_standard_20261007/llama_fullbatch_vs_incremental.json`
- `/tmp/llama_standard_20261007/llama_ubatch1_vs_incremental.json`
- Incremental and full-batch llama l_last dumps: `/tmp/llama_standard_20261007/llama_state/` and `/tmp/llama_standard_20261007/llama_batch_state/`.

## Matched Option-B performance setup

Use `configs/testing/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml` as the baseline config. It right-sizes the static Option-B set to **8,421 resident experts / 10.842 GiB**. Both arms must use this exact set, independent of reported VRAM percentage. The legacy cache was a 6.44 GiB allocation with `0/4650` slots filled; it is dead VRAM, not resident MoE work. The accepted early-ready run can decode around 33.8 tok/s with the legacy cache off and this same 8,421-expert plan at roughly 15.4 GiB peak on the 3090 (~63%). Do not force 90% VRAM allocation just to match an old empty cache.

A prior branch-local, no-logit-dump six-pair A/B on binary hash `b2449d6e...` used the same config and logged 8,421 hot experts per graph, 10.842 GiB, and `0/0` legacy slots. It measured early-ready **17.77 tok/s** versus staged **15.06 tok/s**, `-15.25%`, zero staged wins. This is evidence for that exact build snapshot; rerun after the clean-folder build because the current r7 binary has a different hash.

The 33.8 tok/s accepted baseline was measured with another binary/build. Do not attribute the cross-binary speed gap until both binaries are built against the same GGML source/configuration. The erroneous root-checkout control run from this session is in `/tmp/zeromoe_bugfix_521d67a_20261007/logs/known_good_vs_current_baseline_perf_20261007_r2/`; it compared the wrong root branch and is not valid for this task.

Existing staged-path profiling is documented in `docs/phase2-arena-and-prefix-debug-2026-10-07.md`. The old `ZEROMOE_QWEN4EXP_GRAPH_PHASE_TIMING` switch has been removed; current Qwen layer timing uses `--verbosity 5`. For scheduler split detail use `BEE_GGML_PROFILE=1 BEE_GGML_PROFILE_SPLITS=1`. The retained run-40 profile measured a 69-split tail (34 CPU, 35 GPU): about 39.05 ms/token scheduler time, including 27.05 ms CPU graph compute, versus about 0.40 ms GPU submission and 1.14 ms GPU synchronization. It also recorded 3.76 MB H2D and 0.35 MB D2H. This points to CPU `MUL_MAT_ID` compute/per-split handling as the staged path’s main cost; it does not explain the separate 17.77 tok/s branch baseline versus the older ~33.8 build.

## Reproduction inputs and commands

Machine: Linux, Xeon W-2235, RTX 3090 24 GiB (the tests target GPU 0), RTX 5060 Ti idle; use 10 CPU threads. Model: `/data/sdb/models/qwen3.8-flash-next/Q2_0/`, with first shard `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`. Prompt IDs hash `a2f73504...`; continuation IDs hash `5cfb223b...` (64 IDs). The continuation file in the bug-fix checkout is `logs/phase2_layer3_up_weight_verify_20261007/continuation64.tokens`.

The benchmark script itself sets full-graph decode, prompt token input, teacher forcing, and `ZEROMOE_QWEN_CPU_THREADS=10`. From the bug-fix checkout:

```bash
python3 bench/option_b_early_ready_check.py \
  --engine build_ggmlcuda/zeromoe \
  --out-dir logs/<new-run-id> \
  --prompt-ids logs/quant_decode_compare_chat_20261004/prompt.tokens \
  --continuation logs/phase2_layer3_up_weight_verify_20261007/continuation64.tokens \
  --reps 6 --threads 10 --wait-idle 30 \
  --baseline-env ZEROMOE_QWEN_EARLY_READY_LAYER13=0 \
  --baseline-env ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT=1 \
  --baseline-env ZEROMOE_QWEN_EARLY_READY_RESIDENT_TABLES=1 \
  --baseline-env ZEROMOE_QWEN_EARLY_READY_ASYNC_OUTPUT=1 \
  --candidate-env ZEROMOE_QWEN_EARLY_READY_LAYER13=0 \
  --candidate-env ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT=0 \
  --candidate-env ZEROMOE_QWEN4EXP_PHASE2_EARLY_READY=1 \
  --candidate-env ZEROMOE_QWEN4EXP_STORAGE_SERVICE=1 \
  --candidate-env ZEROMOE_QWEN4EXP_COMPLETION_GRAPH=1 \
  --candidate-env ZEROMOE_QWEN4EXP_COMPLETION_STAGED=1 \
  --candidate-env ZEROMOE_QWEN4EXP_MASTER_GRAPH_STAGED=1 \
  --candidate-env ZEROMOE_QWEN4EXP_PARTIAL_LAYER=13 \
  --candidate-env ZEROMOE_QWEN4EXP_PARTIAL_LAYER_TOP_N=64 \
  --candidate-env ZEROMOE_QWEN4EXP_PARTIAL_LAYER_PHASE2=1
```

Add `--dump-logits` for a correctness run; its timing is invalid. Check actual Option-B hot counts in engine logs, not just the config name. Monitor `nvidia-smi`, `MemAvailable`, and `/proc/vmstat` `oom_kill`; prior full diagnostic runs had zero OOM kills. Keep each run’s manifest, engine logs, logits/hidden dumps, and resource samples.

## Suggested first actions in the next session

1. Start from a clean copy of `bugfix/staged-coherence-521d67a` at `521d67a`, carry over only the relevant uncommitted graph/handoff changes and logs, and exclude EXL3. Do not use `/home/ray/Projects/ZeroMoE` as the task checkout.
2. Configure/build the new folder with the intended GGML source and `-j10`; verify CMake source/root, `ldd`, and executable hash before any inference run.
3. Reproduce branch early-ready versus standard llama on the same fixed history, then compare staged versus both. Use l_last dumps to trace the first baseline mismatch before treating the staged flip as the root cause.
4. After coherence is understood, repeat the no-dump six-pair performance A/B with the exact 8,421-expert Option-B set and enable the phase/scheduler profiling above.

Generated logs and temporary llama harnesses were retained under `/tmp/zeromoe_bugfix_521d67a_20261007/logs/` and `/tmp/llama_standard_20261007/`.
