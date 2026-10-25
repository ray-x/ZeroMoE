# Option-B global pool: numeric validation and design decision

Continuation of [the handoff](handoff-2026-10-03-option-b-global-pool.md).
The changes and validation artifacts remain uncommitted.

## Decision

Keep static GPU-hot / CPU-cold dispatch and leave the shipped/default layouts
unchanged. Global ranking remains experimental. Its equal-VRAM gain in the
handoff was 1.1%, below the user's 1.5% acceptance threshold. The larger
per-layer allocation remains the stronger performance candidate: 12.5% in
the earlier seven-run comparison, at the cost of about 3.7 GiB more VRAM.
Those throughput numbers were not re-measured by the teacher-forced runs.

The tests below found no missing-route or compact-addressing defect in the
tested Q2_0 decode cases. They also confirm that the 48-layer layouts change
the identifier-heavy summary. Passing these tests does not establish broad
quality equivalence, or justify promoting global mode to a default.

## What was fixed

- Startup global selection now uses the sanitized decode `pin_file` records,
  matching the graph planner and VRAM estimate. It previously included
  prefill-exclusive records from the RAM-preload ranking. Before this fix,
  startup logged 76.1356% pooled coverage while the graph logged 76.29%;
  afterward startup logs 76.2949% and the graph logs 76.29%.
- Startup global selection excludes CPU-affinity layers, as the planner does.
- The arena planner no longer applies per-layer `fill_vram` trimming to a
  global pool. Global mode keeps the requested K; it does not silently change
  the allocation through that separate per-layer rule.
- Reported pool coverage includes traffic from layers receiving no hot
  experts when K is smaller than the layer count. A regression test covers
  the clamped floor and zero coverage for those layers.

The normal A/B configurations did not exercise the fill or CPU-affinity
cases. Their original and rebuilt A/B logit dumps were byte-identical;
the startup-ranking correction did not remove the summary discrepancy.

## Teacher-forced comparison

Reference A captures both the exact prompt IDs and a greedy continuation.
Every candidate consumes those same IDs and continuation, computing its own
raw logits before repetition penalties. This removes autoregressive history
divergence from the comparison. KL and total variation use temperature 1;
they are numeric diagnostics, not semantic quality scores.

The problematic prompt is preserved as
[`bench/data/option_b_mid_prompt.txt`](../bench/data/option_b_mid_prompt.txt).
It tokenizes to 402 tokens including the chat template. The run uses the
RTX 3090 at its existing 220 W cap, 10 CPU threads, 64 generated-token
records, and context 2048. Step 0 is the prefill prediction and is included
in the table. Dump I/O and startup costs make these unsuitable throughput
measurements.

| Layout | Raw top-1 agreement with A | Mean KL(A || layout) | Mean teacher-token NLL change |
| --- | ---: | ---: | ---: |
| A repeat | 64/64 | 0 | 0 |
| B: global K=8,421, 48 layers | 61/64 | 0.0216692 | +0.0083280 |
| C: per-layer 256, original 44 layers | 62/64 | 0.0188226 | +0.0166110 |
| D: global K=11,264, 48 layers | 61/64 | 0.0194701 | +0.0165590 |
| V1: global K=8,421, original 44 layers | 62/64 | 0.0046694 | +0.0007413 |
| floor1: global K=48, one expert per layer | 61/64 | 0.0218506 | +0.0135559 |

At step 12, A selects token 48 (`Q`) while B/D select token 10218. A's
top-1/top-2 raw-logit margin is 0.5938, so describing the entire event as
a tiny baseline tie would be inaccurate. The probabilities of token 48 and
token 10218 respectively are:

| Layout | Token 48 | Token 10218 |
| --- | ---: | ---: |
| A | 0.16294 | 0.07358 |
| B | 0.12125 | 0.14367 |
| C | 0.12133 | 0.11683 |
| D | 0.10281 | 0.11744 |
| V1 | 0.26029 | 0.07236 |

C has similar aggregate drift to B/D but stays on the identifier at this
step. Global selection over the original 44 layers has substantially less
drift. This is consistent with sensitivity to which routes execute on CPU
versus GPU; it does not prove the source of every numerical difference.

`ZEROMOE_MMID_NO_NEGATIVE_SKIP=1` makes both branches compute valid dummy
routes and then mask foreign contributions. B and floor1 each produced
**byte-identical full logit dumps** with this control on and off. Negative
skipping therefore does not explain their differences from A in this test.

Artifacts: `logs/option_b_teacher_mid_fixed_20261003/`. Its manifest records
the engine hash, prompt hash, base-config hashes, usage hash and run settings;
`summary.json` and each `*.comparison.json` retain aggregate and per-step
metrics. Exact prompt IDs, continuation IDs, raw logits, generated text and
effective TOML configs are retained alongside them.

A second 64-record continuation uses the ordinary chat prompt preserved in
[`bench/data/option_b_chat_prompt.txt`](../bench/data/option_b_chat_prompt.txt).
The captured A answer is coherent. All candidates replay its exact token
history; those forced texts are not independent quality tests.

| Layout | Raw top-1 agreement with A | Mean KL(A || layout) |
| --- | ---: | ---: |
| B | 63/64 | 0.00132006 |
| C | 63/64 | 0.00080279 |
| D | 63/64 | 0.00079659 |
| V1 | 62/64 | 0.00078662 |

Artifacts: `logs/option_b_teacher_chat_20261003/`. This matrix used a saved
binary in `logs/option_b_validation_engine_20261003/zeromoe`, after the
native build incorporated concurrent state-handoff changes (disabled in
these configs). Each matrix has its own engine hash; comparisons never mix
reference and candidate engines within a matrix.

## Compact tensor regression

[`tests/test_option_b_compact_ggml.cpp`](../tests/test_option_b_compact_ggml.cpp)
uses the real integration GGML backend, Q2_0 weights, D=2560, FF=640 and
top-k=10. It compares a full expert tensor with gathered compact tensors of
1, 2 and 8 experts. Cases cover all-hot, all-cold, alternating routes, one
hot route at the first/middle/last slot, and repeated execution with changing
route IDs. It checks active results against full tensors and inactive results
against zero, also comparing negative skipping with dummy-route masking.

Both CPU and CUDA passed all 54 cases with zero difference from the full
tensor reference. Logs: `logs/option_b_compact_{cpu,cuda}_20261003.log`.
This test does not exercise the engine's scheduler or arena lifetime;
the teacher-forced model controls above exercise those paths.

Build and run it against the same GGML libraries as the engine:

```bash
c++ -std=c++20 -O2 -I../llama.cpp-moestream-integration/ggml/include \
  tests/test_option_b_compact_ggml.cpp \
  -L../moestream-build-upstream-master/ggml-build/src \
  -L../moestream-build-upstream-master/ggml-build/src/ggml-cuda \
  -Wl,-rpath,"$PWD/../moestream-build-upstream-master/ggml-build/src" \
  -Wl,-rpath,"$PWD/../moestream-build-upstream-master/ggml-build/src/ggml-cuda" \
  -lggml-cpu -lggml-cuda -lggml-base -o /tmp/option_b_compact_check
/tmp/option_b_compact_check cpu
# Check nvidia-smi and competing processes before launching the CUDA case.
CUDA_VISIBLE_DEVICES=0 /tmp/option_b_compact_check cuda
```

## Reproduction and required checks

```bash
../moestream-build-upstream-master/gguf_tool verify \
  /data/sdb/models/qwen3.8-flash-next/Q2_0
python3 tests/test_compare_decode_logits.py
make test-components ARGS="option_b_global_pool"
make test BUILD_DIR=../moestream-build-upstream-master JOBS=4
python3 bench/option_b_global_pool_check.py \
  --prompt-file bench/data/option_b_mid_prompt.txt \
  --out-dir logs/option_b_teacher_mid_new \
  --ngen 64 --layouts A_repeat,B,C,D,V1,floor1,B_dummy,floor1_dummy \
  --wait-idle 600
```

The model verifier, component test, five comparator tests and full native
Linux/CUDA suite passed. The initial standalone `make gguf-tool` wrapper
failed to link the unrelated `cmd_mix` change; verification used a temporary
current-source tool including `tools/gguf_mix.cpp`. The complete CMake build
also builds the updated GGUF tool. The Makefile's test recipe requires a
relative `BUILD_DIR` (it prefixes the executable path with `./`).

The correctness runner checks for competing inference before each launch
and can wait for the GPU to become idle. `--resume` requires matching saved
settings and inputs. Use a saved engine binary if another session is building
in the shared build directory; the runner rejects an engine change during
a matrix. Concurrent state-handoff work in the shared tree was preserved.

`bench/option_b_global_pool_ab.sh` was also run end to end with one warmup,
one measured repetition and 64 generated tokens per layout (10 CPU threads).
All four generated short answers were coherent. The smoke rows were A 31.76,
B 32.68, C 35.84 and D 36.46 tok/s; one repetition cannot establish the
acceptance threshold. Log: `logs/option_b_global_ab_smoke_20261003.log`, texts:
`logs/optb_global_ab_{A,B,C,D}_1.stdout`. The script now stops on failed or
incomplete runs and checks competing GPU processes before each launch.

The next performance decision needs an order-balanced comparison of the
larger per-layer allocation on IQ1_M and a separate capacity calculation for
the 5060 Ti. Further quality work should use representative prompts and
teacher-forced controls; the saturated exact-identifier copy test should
remain retired. Do not tune layer membership just to reproduce this one
summary.

For a standalone comparison:

```bash
python3 bench/compare_decode_logits.py reference.logits candidate.logits \
  --tokens reference.tokens --steps 64 --json comparison.json
```
