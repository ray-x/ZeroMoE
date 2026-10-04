# Qwen graph refactor: recorded state

This file records the graph worktree state before the scoped commit. The
repository also contains unrelated ExLlamaV3, CPU, configuration, test and
generated-artifact changes; those remain outside the graph commit.

The scoped commit is `54c9c64de668f472eb55b8e97b880aaa799b2f0a`
(`Refactor Qwen graph scheduler overlap`, author `ray-x`).

## Selected runtime

The selected Q2_0 RTX 3090 profile is Option-B 0/44/4:

- 8,421 usage-selected experts remain resident in 10.84 GiB of VRAM.
- Layers 0, 1, 2 and 22 use the plain CPU path.
- Ten CPU threads are used while other host work may be present.
- BF16 MMVF routing remains disabled for this A/B.
- SSD/GDS integration and dynamic staged-GPU execution are disabled.

Source `configs/qwen4exp_q2_0_option_b_scheduler.env` together with
`configs/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml` when using the
patched `../moestream-build-upstream-master` binary.

## Graph changes in this commit

The early-ready scheduler path gives each split layer a private CPU cold-expert
graph. An input-ready event starts that graph from the scheduler integration
hook while already-submitted GPU work proceeds. The path keeps stable private
input/output buffers and does not reconstruct the main graph per token.

Resident remap and mask tables are uploaded once into owned GPU buffers. The
CPU graph retains its host tables, while the main GPU graph reads immutable
device copies. Cold output can use an owned pinned host buffer and an upload
stream joined by an event before the merge.

The experimental epoch-tagged staging pool is retained behind flags for
correctness research. It publishes all three quantized planes only after their
copies complete, validates the epoch and expert ID on the GPU, holds banks
through the consuming token, and falls back to CPU when publication is late.
It is not enabled by the selected profile.

The CUDA conditional graph remains an optional pure-GPU all-or-nothing path.
Conditional bodies contain no host callbacks. The current early-ready path is
the accepted scheduler design; the conditional path is not combined with a
host callback in one child graph.

## Measurements

The final six-permutation, order-balanced decode comparison used the same
binary, model configuration, forced 64-token history and ten CPU threads:

| Mode | tok/s | Result |
| --- | ---: | --- |
| Original graph/scheduler | 31.848 | baseline |
| Early-ready + resident tables | 33.294 | +4.538%, 6/6 wins |
| Above + asynchronous output | 33.874 | +6.361% overall, +1.744% incremental, 6/6 wins |

Evidence is in `logs/option_b_scheduler_final_three_arm/summary.json`.
The current-binary follow-up in `logs/option_b_scheduler_post_native_logits/`
and the original parity run in
`logs/option_b_scheduler_final_three_arm_logits/` contain byte-identical
64-record logits for all three modes, with zero KL, TV and maximum delta.

To reproduce the three-arm measurement, first verify the Q2_0 directory, make
sure the host and GPU are idle, then run:

```bash
python3 bench/option_b_early_ready_check.py \
  --engine ../moestream-build-upstream-master/moestream \
  --out-dir logs/option_b_scheduler_repro \
  --prompt-ids logs/quant_decode_compare_chat_20261004/prompt.tokens \
  --continuation logs/quant_decode_compare_chat_20261004/Q2.tokens \
  --reps 6 --all-split --wait-idle 300 \
  --candidate-env MOESTREAM_QWEN_EARLY_READY_RESIDENT_TABLES=1 \
  --third-env MOESTREAM_QWEN_EARLY_READY_RESIDENT_TABLES=1 \
  --third-env MOESTREAM_QWEN_EARLY_READY_ASYNC_OUTPUT=1
```

The three arms are the original scheduler, early-ready plus resident tables,
and that bundle plus asynchronous output. The script records admission state,
rejects compiler/inference interference, and writes `summary.json` with the
paired gains. For a normal run rather than an A/B, source the runtime overlay
and launch the configured engine:

```bash
source configs/qwen4exp_q2_0_option_b_scheduler.env
../moestream-build-upstream-master/moestream \
  --config configs/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml
```

The rejected experiments remain recorded:

- all-hot CPU skip: +0.029%, 2/6 wins;
- predictor alone: −0.049%, 3/6 wins;
- Layer-13 staging: −0.432%, 2/6 wins;
- all-layer staging: −12.636%, 0/6 wins.

The all-layer verifier nevertheless passed 1,344 independent staged-output
checks at the diagnostic boundary, with maximum delta zero. The strict worker
delay test produced byte-identical fallback logits and zero staged routes.
Three natural-stop requests (code, creative, code) completed in one process;
the first and third code answers matched before and after the creative request.

The native suite passed with:

```text
CUDA_VISIBLE_DEVICES=0 make test BUILD_DIR=../moestream-build-upstream-master JOBS=4
```

The retained log is `logs/option_b_scheduler_final_native.log`.

## Milestone 1: immutable route classification

The first follow-up improvement adds `moe_partial::RouteClassification` to
`include/moe_partial_route.hpp`. It creates an immutable assignment for each
original router slot, carrying the expert ID, original slot, router weight,
execution target, branch-local weight slot and slot generation. Resident,
staged and CPU ownership are mutually exclusive; a CPU assignment is the
explicit fallback when no published staged slot exists. Duplicate expert IDs
remain separate route slots, so their router weights cannot be accidentally
collapsed.

The classifier currently sits beside the graph code and is not yet wired into
the accepted decode path. That keeps the production baseline unchanged while
the exact-once contract is tested before adding conditional or host-node
branches.

The focused test covers all-hot, all-cold, mixed, duplicate-route, overlapping
ownership, invalid IDs, generation propagation and both negative-sentinel and
dummy-route modes:

```bash
g++ -std=c++23 -O2 -Iinclude tests/test_moe_partial_route.cpp \
  -o /tmp/test_moe_partial_route_route_classification
for mode in capable stock optout; do
  /tmp/test_moe_partial_route_route_classification "$mode"
done
```

All three modes passed. The next milestone can consume this descriptor from a
DDR-only partial-layer graph without mutating shared remap or mask tables.

## Scope and reproducibility

The scoped graph files are:

- `src/ggml_qwen4exp_graph.cpp`
- `cuda/qwen4exp_master_graph.cu`
- `include/qwen4exp_early_ready.hpp`
- `include/qwen4exp_stage_pool.hpp`
- `include/option_b_global_hot.hpp`
- `patches/ggml-scheduler-early-ready-layer13.patch`
- `bench/option_b_early_ready_check.py`
- `bench/analyze_qwen_route_prediction.py`
- `bench/option_b_topic_check.py`
- `bench/option_b_request_check.py`
- `configs/qwen4exp_q2_0_option_b_scheduler.env`
- the graph handoff and worklog documents

The patch filename retains its historical Layer-13 name; the scheduler hook
now supports all eligible split layers. The commit does not include generated
logs, build directories, unrelated README changes, or unrelated ExLlamaV3 and
native-test edits. Those files remain in the worktree for their owners.
