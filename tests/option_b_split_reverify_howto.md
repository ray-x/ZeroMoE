# Option-B split-layer decode testing: how to reproduce

How to build an isolated test environment and re-run the Option-B hot/cold
split-layer decode benchmarks (the "X/split/plainCPU" configs, e.g.
`0/44/4`) without touching the shared working tree — useful when another
session has in-progress uncommitted edits in the main checkout.

## 1. Create an isolated worktree

```bash
git worktree add --detach /tmp/reverify_wt <commit-or-branch>
```

Two files this repo keeps untracked but CMake needs are not copied by a
fresh worktree checkout — copy them in manually:

```bash
cp CMakeLists.txt Makefile /tmp/reverify_wt/
```

If you're testing against a regenerated MoE usage histogram (decode
router-usage tap), that file is also untracked — copy it in too:

```bash
mkdir -p /tmp/reverify_wt/tests/moe_usage_survey
cp tests/moe_usage_survey/moestream_usage_survey_qwen4exp_usage_tap /tmp/reverify_wt/tests/moe_usage_survey/
cp tests/moe_usage_survey/moestream_usage_survey_prefill /tmp/reverify_wt/tests/moe_usage_survey/
```

## 2. Build

**Both** of these CMake flags are required for the CUDA-graph decode path —
`MOESTREAM_USE_GGML=ON` alone silently builds without ggml's own CUDA
backend, and `Test 34` (GLM5NEXT KDA dense projection fusion) will report
"Skipped (GGML CUDA backend not compiled)" as the tell:

```bash
cmake -S /tmp/reverify_wt -B /tmp/reverify_wt/build_ggmlcuda \
  -DCMAKE_BUILD_TYPE=Release -DMOESTREAM_USE_GGML=ON -DMOESTREAM_USE_GGML_CUDA=ON \
  -DMOESTREAM_GGML_ROOT=/home/ray/Projects/beellama.cpp/ggml
make -j$(nproc) -C /tmp/reverify_wt/build_ggmlcuda
```

A from-scratch `ggml-cuda` build compiles a large number of
`fattn-mma-kvarn-instance-*.cu` / `fattn-vec-instance-*.cu` flash-attention
template files — this legitimately pins most CPU cores for several minutes
(load average ~24 is normal, not a hang). Run it via a background task and
poll `[ X%] Building CUDA object ...` progress lines rather than waiting
synchronously.

## 3. Sanity-check the build

```bash
/tmp/reverify_wt/build_ggmlcuda/moestream_tests
```

Look for `✅ All core engine tests passed successfully!` and confirm no
tests unexpectedly report `Skipped (GGML CUDA backend not compiled)`.

## 4. Config file

`device_expert_layers` (the trailing full-GPU-resident layer-block size —
the `X` in `X/split/plainCPU` naming) has **no TOML key**; it is env-var
only (`MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_LAYERS` /
`MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1`). Everything else about the
split lives in the TOML. See `configs/qwen4exp_option_b_rightsize_split44_x0.toml`
in the main repo for a complete worked example; the load-bearing keys are:

```toml
model_path = "/data/sdb/models/qwen3.8-flash-next/UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf"
pin_file = "tests/moe_usage_survey/moestream_usage_survey_qwen4exp_usage_tap"
prefill_pin_file = "tests/moe_usage_survey/moestream_usage_survey_prefill"
freeze_usage_stats = true

qwen4exp_option_b_selected_layers = [3, 4, 5, ...]   # the split-layer list
qwen4exp_option_b_top_n = 256
qwen4exp_option_b_min_coverage = 0.80                # recalibrate to match real concentration -- see step 6
qwen4exp_option_b_right_size_top_n = true

qwen4exp_full_graph = true
qwen4exp_full_graph_decode = true
# ... plus the standard dense/GDN/attention GPU flags and one [[gpus]] block
```

`pin_file`/`prefill_pin_file` are relative paths — the process must be run
with its working directory set to the worktree root (or wherever those
paths resolve from).

## 5. Run

```bash
cd /tmp/reverify_wt
echo "Explain what a binary search tree is and give a short example in Python." | \
MOESTREAM_QWEN4EXP_FULL_GRAPH=1 MOESTREAM_QWEN4EXP_FULL_GRAPH_DECODE=1 \
MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_LAYERS=0 MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1 \
MOESTREAM_BENCH_IGNORE_STOP=1 CUDA_VISIBLE_DEVICES=0 MOESTREAM_CUDA_SCHEDULE=auto \
./build_ggmlcuda/moestream --config path/to/config.toml --ngen 40
```

`MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_LAYERS` is the `X` for whichever row
you're testing (e.g. `0` for the `0/44/4` row).

Before launching, always check the GPU is actually idle
(`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` and
`pgrep -af moestream`) — this box runs concurrent sessions, and a leftover
or unrelated process silently starves the run instead of failing loudly.

## 6. Recalibrating `qwen4exp_option_b_min_coverage`

`min_coverage` is the per-layer target coverage that both (a) rejects a
layer back to plain-CPU dispatch if even the full `top_n` ceiling can't
reach it, and (b) (with `right_size_top_n=true`) sizes each layer's hot-set
down to the smallest N that reaches it. Whether 0.90 or 0.80 (or something
else) is the right value depends on how concentrated the *current* real
usage histogram actually is — recompute before assuming the old value still
holds:

```python
import collections
layer_counts = collections.defaultdict(dict)
with open("tests/moe_usage_survey/moestream_usage_survey_qwen4exp_usage_tap") as f:
    for line in f:
        layer, expert, cnt = map(int, line.split())
        layer_counts[layer][expert] = cnt

def experts_needed_for(counts, target):
    total = sum(counts.values())
    sorted_c = sorted(counts.values(), reverse=True)
    cum = 0
    for i, c in enumerate(sorted_c):
        cum += c
        if cum >= target * total:
            return i + 1
    return len(sorted_c)

scores = {l: experts_needed_for(c, 0.80) for l, c in layer_counts.items()}
avg = sum(scores.values()) / len(scores)
print(f"avg experts needed for 80% coverage: {avg:.1f} / 512 ({avg/512*100:.1f}%)")
```

Rank layers by this score (ascending = most concentrated) to build
`qwen4exp_option_b_selected_layers` for a given split-layer count `N` and
trailing-GPU-block size `X`: the pool is layers `[0, n_layers - X)` (layers
already inside the trailing GPU-resident block are never split candidates),
ranked, and the top `N` selected.

**A real, measured tradeoff, not just a data-recalibration exercise:**
lowering `min_coverage` shrinks each split layer's GPU hot-set (cheaper GPU
side, and lets the requested `X` fit without the engine's internal
VRAM-shrink retry kicking in) but pushes more of that layer's real routed
traffic onto the CPU cold path. At low-to-moderate split-layer counts
(32-36 of 48) this nets out roughly neutral; at high split-layer counts
(42-44, near the whole model) the CPU-side cost increase measurably
outweighs the GPU-side savings — a real ~11% throughput drop was measured
going from `min_coverage=0.90` to `0.80` at 42-44 split layers, despite 0.80
being the technically-correct recalibration for a less-concentrated usage
histogram. Don't assume the "correct" threshold is free.

## 7. Extracting steady-state tok/s

The first 1-2 decode tokens after prefill pay a one-time warmup cost
(CUDA-graph capture/launch residuals, cache warmup) that is not
representative of sustained throughput — exclude them:

```bash
grep "GGML-QWEN-FULL-DECODE-TIME" your_log.log | grep -o "total=[0-9.]* ms" | grep -o "[0-9.]*" | tail -n +3 | \
  awk '{s+=$1;n++} END{printf "mean=%.3fms -> %.2f tok/s (n=%d)\n", s/n, 1000.0/(s/n), n}'
```

If the engine's internal VRAM-retry mechanism had to shrink
`device_expert_layers` before landing on a working config (logged as
`retrying with device_expert_layers shrunk by N`), extract only the
**last** attempt's block before computing steady-state — earlier failed
attempts' `[OPTION-B] ... static hot/cold split` and decode-time lines are
still in the log and will double-count otherwise:

```bash
awk '/retrying with device_expert_layers shrunk/{buf=""} {buf=buf"\n"$0} END{print buf}' your_log.log
```

## 8. Always run 3+ reps, and check coherence before trusting throughput

Single runs on this box have shown 1-3 tok/s swings between otherwise
byte-identical configs and launch commands — a cause not yet fully isolated
(CPU governor and run-order were both tested and ruled out). Never trust or
compare a single run's tok/s number without first reading the actual
generated text (`grep -A5 "Output:" your_run.stdout`) — a broken decode
path can produce a *faster*, not slower, number.

## 9. Reproduced 2026-10-02 on the upstream integration ggml (and a regression to check first)

Build used: `MOESTREAM_GGML_ROOT=/home/ray/Projects/llama.cpp-moestream-integration/ggml`
(not beellama's ggml as in step 2), `build dir ../moestream-build-upstream-master`.
Result for `configs/qwen4exp_option_b_rightsize_split44_x0_baseline.toml`, prompt
"write a python program to print hello world", `--ngen 40`: **34.9-35.8 tok/s**
(8+ runs), same output text as the recorded baseline; hot-set hit rate ~90%.

**Required: the linked ggml must export `ggml_cpu_mul_mat_id_negative_skip()`.**
`include/mmid_route_skip.hpp` weak-links it. If it is missing, every split layer silently
falls back to a dummy expert id 0 + mask and both the CPU and GPU compute all top-k routes:
0/44/4 drops from ~35 to ~19 tok/s with an otherwise identical log (same splits, same hit
rate). The function lives in `patches/ggml-mul-mat-id-negative-skip.patch` (ggml-cpu.c).
CMake skips the patch when the `moestream_mmid_negative_skip` marker is already in `mmvq.cu`,
so an already-patched tree needs the function added by hand. Check with:

```bash
nm -D <build>/ggml-build/src/libggml-cpu.so | grep negative_skip   # must print a T symbol
```

Tell-tale of the broken state: `MOESTREAM_SCHED_PROF=1` (env-gated profiler in the integration
ggml's `ggml-backend.cpp`) shows ~37 ms/token of "CPU compute" over 49 CPU splits and
`perf` shows the IQ dot kernels at ~35% of samples; healthy is ~3.5% (libgomp idle spin ~90%).

**Things that do NOT help 0/44/4 (measured, order-balanced A/B, 4 runs per side):**
- `OMP_WAIT_POLICY=active GOMP_SPINCOUNT=10000000`: 35.33 vs 35.32 tok/s.
- Decode-only HC stream-mean fusion (permute+cont+mean): 35.45 vs 35.66 (+0.6%, noise).
- `MOESTREAM_QWEN4EXP_DECODE_RECREATE_GRAPH_ARENA=1`: explicitly restores the
  older separate-arena behavior. The default is to borrow the completed
  prefill graph's dense/expert arenas; split configurations should only use
  this override when arena borrowing is incompatible with the selected split.
- CPU kernel rewrites: cold-path kernels are ~3.5% of samples here, so there is nothing to win.

Bisect recipe if the number drops again: run an older `build/moestream` on the same config
(isolates machine vs code), then cross-load libs with `LD_LIBRARY_PATH` (beats RUNPATH) to split
binary vs ggml libs, then `git checkout <commit>~1 -- <files>` for the suspect commit.

## 10. CUDA conditional node (Phase 2 master graph): what it takes and why it does not engage (2026-10-02)

The conditional node is `MOESTREAM_QWEN4EXP_MASTER_GRAPH=1` on top of Phase 2
(`MOESTREAM_QWEN4EXP_PARTIAL_LAYER=<layer>` + `MOESTREAM_QWEN4EXP_PARTIAL_LAYER_PHASE2=1`). Per token the
device-side predicate checks the bitmap of VRAM-resident experts of the split layer and a
`cudaGraphCondTypeIf` node runs the GPU-only suffix (`sgpu`) only if every routed expert is resident.

Findings with `host-node` CPU layers (`MOESTREAM_QWEN4EXP_HOST_NODE=1`, `qwen4exp_cpu_offload_layers` = every CPU layer),
22 trailing GPU layers, partial layer 13:
- Without host nodes the safety gate refuses it (`n_splits=28/25`: CPU layers are scheduler splits).
- With host nodes the gate passes (`splits=1`, `[QWEN4EXP-MASTER-GRAPH] enabled`), output identical on/off,
  but throughput is unchanged (17.69 vs 17.68 tok/s, order-balanced) because **the master graph is never built**:
  1. `sgpu` only executes on tokens whose routed experts are all hot, so ggml never captures it
     (`moestream_ggml_cuda_get_captured_graph` returns null). `MOESTREAM_GGML_CUDA_GRAPH_CAPTURE_ON_FIRST_CALL=1`
     captures it on its first execution and gets both handles.
  2. Then `cudaGraphInstantiate` fails with `invalid argument`: **`sgpu` contains 12 host nodes** (the CPU layers 14-25
     are `cudaLaunchHostFunc` callbacks) and CUDA does not allow host nodes in a conditional body graph. Not fixable by config.
- Fixed: a failed build step no longer leaves a sticky CUDA error (it used to abort the engine on the next kernel
  launch) and is no longer retried every token; it logs one `[QWEN4EXP-MASTER-GRAPH] disabled for this session` warning.
- `MOESTREAM_QWEN4EXP_GRAPH_REPLAY_TIMING=1` crashes the Phase 2 host-node path (`CUDA error: invalid argument`, event
  timing inside captured graphs); use `MOESTREAM_TOKEN_LOOP_DIAG=1` for timing.
- To make a conditional body valid, it must contain only kernels/memcpys (e.g. just the split layer's GPU hot-expert MoE),
  with the CPU host node left outside the body and gated by a flag the predicate writes.

## 11. Split-layer progression, upstream integration ggml, fixed build (2026-10-02)

"Conditional node" in the user's sense = an Option-B split layer (hot experts resident in VRAM, cold on CPU, per-layer
hot/cold route tables), grown from one layer to 44. Same prompt/method as section 9 (`--ngen 40`, warm-up + 2 reps):

| config | GPU layers / split / plain CPU | tok/s (rep1, rep2) |
|---|---|---|
| plain 22/26 full graph (`mixed_dl22.toml`, no Option B) | 22 / 0 / 26 | 26.25, 25.37 (80 tok) |
| `qwen4exp_option_b_layer13.toml` | 22 / 1 / 25 | 26.93, 27.31 |
| `..._split30_x7.toml` | 7 / 30 / 11 | 30.66, 29.26 |
| `..._split32_x6.toml` | 6 / 32 / 10 | 31.38, 31.68 |
| `..._split36_x4.toml` | 4 / 36 / 8 | 31.63, 31.65 |
| `..._split42_x1.toml` | 1 / 42 / 5 | 31.23, 32.54 |
| `..._split44_x0_baseline.toml` | 0 / 44 / 4 | 34.65, 35.57 |

One split layer has no effect on speed; the gain grows with the number of split layers. The earlier ~38 tok/s figure was not
reproduced (35-36 here; memory note `split44-cov80-baseline` says that goal needed removed hacks).
Do not confuse this with the CUDA conditional graph node of section 10 (Phase 2 master graph), a different mechanism.
Note: running ALL CPU layers as host-callback nodes (`MOESTREAM_QWEN4EXP_HOST_NODE=1`) is much slower (18.3 vs 26 tok/s on 22/26):
callbacks take 39.7 ms/token vs ~25 ms of scheduler-split CPU compute for the same layers (perf: dot-kernel samples equal,
the extra time is non-kernel overhead); that explains the 17.7 tok/s of the section-10 experiment, not the conditional node.

## 12. GPU staging of host-resident MoE layers (`qwen4exp_gpu_stage_enabled`), 2026-10-02

Config = plain 22/26 full graph (`configs/cpu_affinity_test_tmp/mixed_dl22.toml`) + `qwen4exp_gpu_stage_enabled = true` and
`qwen4exp_gpu_stage_layers = [...]` (root keys, before the first `[[gpus]]`). Per token a host callback gathers the layer's
top-10 routed experts (~16 MB) from the DDR arena into a compact VRAM buffer, then ordinary GPU MUL_MAT_ID computes them.
Prompt "What is the meaning of life?", 80 tokens, `MOESTREAM_TOKEN_LOOP_DIAG=1`:

| staged layers | ms/token | tok/s |
|---|---|---|
| none (plain CPU for those layers) | 38.1 | 26.25 |
| 1 (layer 13) | 41.4 | 24.18 |
| 6 (10-15) | 58.8 | 16.99 |
| 13 (even layers 0-24) | 79.3 | 12.61 |
| 26 (all CPU layers) | 112.4 | 8.90 |

Output coherent for layer 13. Cost is linear, ~3 ms per staged layer (callback body ~1.9 ms host gather + ~1.3 ms H2D) vs
~1 ms for the CPU to compute the same layer, so staging ALL top-k experts is a net loss. It would only win if it staged just
the cold (non-resident) experts of an Option-B split layer; today a staged layer replaces the whole layer (staging has
precedence over the split and over CPU-offload in `build_moe`). Timing events inside the captured graph fail
(`event timing unavailable ... invalid argument`), same limitation as section 10.

## 13. 0/44/4 per-token breakdown and the all-hot upper bound (2026-10-02, machine idle, fixed build)

`MOESTREAM_SCHED_PROF=1` (env-gated host-side profiler in the integration ggml's `ggml-backend.cpp`), 300 tokens, steady window =
differenced cumulative averages (graphs 128-288): CPU-split waits+input copy 13.2 ms, CPU compute 15.6 ms (49 splits, ~0.32 ms
each), wait/copy before GPU 2.2 ms, GPU launch 0.55 ms, GPU tail 1.3 ms, total ~32.8 ms/token (iter_total 35.3 ms at this context length).
Routing statistics (200 tokens, 8,800 split-layer visits; valid regardless of load): all 10 routed experts hot in 29.6% of visits;
1 cold 25.9%, 2 cold 18.1%, 3 cold 12.4%, 4+ cold 14.0%; mean 1.69 cold experts per split layer.
CPU compute per cold expert works out to ~0.21 ms vs ~0.05 ms of real dot-product work, i.e. each CPU split is mostly fixed
overhead (fork-join + barriers over ~4 nodes). Upper bound for skipping the CPU split on all-hot layers: ~0.3 x 44 x ~0.25 ms = ~3 ms/token (~8-10%),
and a CUDA conditional node cannot skip a ggml scheduler CPU split (separate backend); host-node callbacks were measured slower (section 11 note).
Do not take profiler numbers while another session compiles (nvcc/ptxas with -j12): the same run then reads 2-3x slower.

## 14. CPU-split overhead investigation (2026-10-02) - correction of section 13's "fixed overhead" guess

Standalone benchmark of one split's graph (3x MUL_MAT_ID gate/up IQ1_M + down IQ4_NL + swiglu, 10 slots, N valid; ggml CPU backend, OpenMP):
- Fixed overhead is SMALL: 1 trivial node ~5 us, an all-skipped split (0 valid ids) ~40 us at 12 threads. Section 13's "~12 ms/token of fixed
  per-split overhead" was wrong (the per-op layer timer under-counts; do not use it to size CPU work).
- The ~0.32 ms/split in the engine is real expert math at poor thread scaling: 2 valid experts = 868 us on 1 thread, 397 us on 12 threads
  (ideal on 6 cores ~125 us); 10 valid = 3837 -> 840 us. Single-thread speed matches the dot kernels (~3.8 GB/s/core).
- In the tight CPU-only loop passive OpenMP wait is 27% faster (286 vs 395 us) and fused gate+up 12% faster (346 us), BUT neither carries over:
  end-to-end on 0/44/4 (3 reps x 5 settings, interleaved, machine idle): default 34.31 tok/s, passive 30.43 (-11%), GOMP_SPINCOUNT=1000
  32.27 (-6%), fused gate+up (`MOESTREAM_QWEN4EXP_GRAPH_MERGED_GATE_UP=1`) 34.88 (+1.7%, noise), passive+fused 31.27 (-9%).
  Threads idle during GPU phases between splits, so waking them every split costs more than spinning saves. Keep spin (default).
- Remaining headroom is ~3x thread-scaling inefficiency inside mul_mat_id for 1-3 experts (barrier-separated short phases); not a config change.
- Process hygiene: another session's `llama-bench`/compile on this box inflated or blocked runs three times; check
  `nvidia-smi` + `pgrep -x nvcc|llama-bench` first, and never `pgrep -f` for patterns that appear in your own command line.

## 15. FIX: serial activation quantization in CPU MUL_MAT_ID (+4% decode), 2026-10-02

Root cause of the poor thread scaling of tiny CPU `MUL_MAT_ID` calls (decode, 1-3 valid experts): `ggml_compute_forward_mul_mat_id` split the
activation quantization (`from_float`, ~2.5k floats) into per-thread block ranges and then every thread read the whole buffer. That costs far more
than the quantization (standalone benchmark `bench/mmid_decode_scaling_bench.cpp`: gate-shaped op, 2 valid experts, 12 threads: 172 us before,
57 us after; `down` 70 -> 54 us). A plain OpenMP loop over the same weights/kernels scaled 3.8-7.7x, ggml's op 1.4-1.6x; adding only the parallel
quantization + barrier to the plain loop reproduced the loss, and quantizing on thread 0 alone (26 us) removed it.
Change (integration ggml `ggml-cpu.c`, also in `patches/ggml-mul-mat-id-negative-skip.patch`): when `ggml_nelements(src1) <= 16384` and nth > 1,
thread 0 quantizes all of src1; the existing barrier after the row grouping publishes it. `MOESTREAM_MMID_PARALLEL_QUANT=1` restores the old path.
- Exact: output bit-identical old vs new at 1/4/12 threads for IQ1_M, IQ4_NL (ne11=10), Q8_0, IQ2_XXS (`bench/mmid_exact_check.cpp`); end-to-end output hash identical.
- 0/44/4 (hello-world prompt, `--ngen 40`, 4 interleaved pairs, idle machine): old 35.49 (+ one 31.43 outlier) -> new 37.00 tok/s (36.77-37.15): +4.0% (+7.3% with the outlier).
- Plain 22/26 (80 tokens, 3 interleaved pairs): old 26.91 -> new 27.95 tok/s: +3.9%; every new run beat its paired old run.
- Prefill is unaffected (src1 larger than the threshold keeps the per-thread path). Spin policy still must stay default (section 14).

## 16. Host-node mode re-measured: the MUL_MAT_ID fix does not apply, but a thread-contention fix does (+24%), 2026-10-02

Config: plain 22/26 + `qwen4exp_cpu_offload = true` for all 26 CPU layers + `MOESTREAM_QWEN4EXP_HOST_NODE=1` (CPU MoE as CUDA host callbacks),
80-token prompt, `MOESTREAM_TOKEN_LOOP_DIAG=1`, idle machine, interleaved reps.
- Serial-quantization fix (section 15) on host-node mode: 18.27 -> 18.41 tok/s (+0.8%, noise). Host-node was 66% of plain speed (28.00).
- Cause of the slowness: the CPU callback runs on a CUDA driver thread as OpenMP team master while the generation thread spins in
  `cudaStreamSynchronize` (CUDA schedule Auto = spin), so ~13 runnable threads share 12 logical CPUs and stragglers stall every barrier.
  In plain mode the generation thread is inside the OpenMP team, so this does not occur.
- Fix tested (3 interleaved reps each): default 18.18 tok/s | `MOESTREAM_CUDA_SCHEDULE=blocking` 22.63 (+24.4%) | `MOESTREAM_QWEN_CPU_THREADS=11` 22.10 (+21.6%) |
  both 22.03 (not additive: each frees a CPU for the waiting thread). Host-node is now ~81% of plain speed.
- Made automatic: `include/gpu_topology.hpp` now defaults the CUDA schedule to blocking when `MOESTREAM_QWEN4EXP_HOST_NODE=1` and
  `MOESTREAM_CUDA_SCHEDULE` is unset/empty (explicit setting still wins; verified log `[CUDA-SCHEDULE] mode=blocking`, 22.5 tok/s; plain mode stays `auto`).
- Earlier host-node-based experiments in this document (section 10 Phase 2 / master graph at 17.7 tok/s, section 11 note) ran with the spinning schedule
  and were handicapped by this; the conditional-node blocker (host nodes cannot live in a conditional body) is unchanged.
- Remaining host-node gap vs plain: ~5 ms/token (22.6 vs 28.0 tok/s), cause not yet isolated (callback arming/guard, per-layer copies, the per-expert add chain).

## 17. Q2_0 (VNNI) measurements and a default-change regression, 2026-10-02 (evening)

Q2_0 (`/data/sdb/models/qwen3.8-flash-next/Q2_0`, 37.6 GB), upstream integration ggml with the Q2_0 VNNI dot kernels (`vpdpbusd` present in `ggml_vec_dot_q2_0_q8_0`),
hello-world prompt, `--ngen 40`, current tree, `MOESTREAM_QWEN4EXP_DECODE_RECREATE_GRAPH_ARENA=1` (see warning below):
- All 48 layers' experts on the CPU (scratch config = `qwen4exp_q2_0_fullgraph_22_26.toml` with `cpu_affinity_layers` = 0..47, `PRELOAD_LAYERS=0`): **18.81 tok/s** (53 ms/token), coherent. CPU-only > 10 tok/s confirmed.
- Plain 22 GPU / 26 CPU (`qwen4exp_q2_0_fullgraph_22_26.toml`): 27.74 tok/s.
- Option B 0/44/4 (`qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml`, hot=256 cold=256): **32.03 tok/s**, coherent; the older "Q2_0 + Option-B DDR staging bug" does not reproduce.
- IQ1_M for comparison: 22/26 28.0, 0/44/4 37.0 tok/s.
- Overlap analysis (section on #5) was on IQ1_M; with Q2_0 the CPU cost per full 10-expert layer is ~0.8 ms (53 ms - ~14 ms GPU over 48 layers), similar to IQ1_M's ~0.95 ms, so the conclusion is unchanged.

WARNING (uncommitted working-tree change, not from this document's author): `src/ggml_qwen4exp_graph.cpp` currently makes arena borrow and bootstrap
the DEFAULT (`MOESTREAM_QWEN4EXP_DECODE_RECREATE_GRAPH_ARENA=1` opts out; `MOESTREAM_QWEN4EXP_GRAPH_BOOTSTRAP=0` disables bootstrap). With that default every per-expert split config
fails: `borrowed-arena graph build failed ... using layered decode` -> 0/44/4 drops from 36.5-37 to **1.12 tok/s with garbled output**
("Here the was delivered a print thew..."), and an all-CPU config (graph_expert_layers=0) fails the same way (3.89 tok/s, garbled). Setting
`MOESTREAM_QWEN4EXP_DECODE_RECREATE_GRAPH_ARENA=1` restores 36.54 tok/s coherent. Needed fix before committing that default: when the borrowed build fails,
retry with freshly built arenas (recreate) instead of falling to layered decode, or skip borrowing when a per-expert split / zero device-expert layers is planned.

## 18. Decode graph construction < 1 s for split configs (2026-10-02, night)

Target: decode graph setup (`[GGML-QWEN-FULL-DECODE-SETUP] prepare_total`, first decode token) under 1 s. Measured on the final build, default settings:
IQ1_M 0/44/4 **100.9 ms** (was 3229 ms with `RECREATE_GRAPH_ARENA=1`, and BROKEN/garbled with the new borrow default), Q2_0 0/44/4 **96.7 ms** (was broken), IQ1_M 22/26 131.7 ms, Q2_0 22/26 129.3 ms,
Q2_0 all-CPU 131.0 ms (was broken). Output coherent everywhere; throughput unchanged (IQ1_M 0/44/4 36.9, Q2_0 0/44/4 32.1, Q2_0 22/26 27.9, Q2_0 all-CPU 18.3 tok/s).
Where the old 3.2 s went: building the per-expert split tensors (`split_calls` 2.65 s: 256 hot experts x 3 projections x 44 layers, ~13.3 GiB gathered from DDR and uploaded), plus ~0.4 s dense upload.
13.3 GiB over PCIe 3.0 x16 is >= 1.1 s on its own, so it cannot fit in the post-prefill path; it has to be uploaded at startup and stay resident.
Changes (all in `src/ggml_qwen4exp_graph.cpp`):
- Decode borrow no longer fails on split / zero-device-layer configs: it borrows only what an owner holds (dense from the prefill owner; experts from the prefill owner only if it has an expert arena, else from the new decode arena owner, else fresh). If a borrowed build still fails it now rebuilds with fresh arenas instead of falling to the layered path (which produced garbled output).
- Bootstrap (split configs only) builds the decode graph once at startup (+~2.7 s one-time startup, VRAM left free ~2.1 GiB) and keeps just its expert arena + address map in `g_graph_decode_arena_owner` (added to `qwen4exp_full_graph_shutdown`); the real decode graph rebinds those device addresses with no upload.
- Prefill borrows the retained dense arena when this graph has no resident expert layers (saves a second 3.3 GiB dense arena and ~400 ms in prefill).
- Per-prompt VRAM policy (`manage_decode_arena_for_prefill`): prefill workspace is ~0.5 GiB + 7.5 MiB/token (measured 0.69 GiB at S=20, 13.18 GiB at S=1747), which cannot coexist with the 13.3 GiB arena. Short prompts (roughly <= ~120 tokens on a 24 GiB card) keep the arena resident; a longer prompt releases it first (that request pays the old ~3.1-3.6 s decode build), and the arena is re-retained from the live decode graph when the next short prompt arrives.
Verified sequence in one session (short, 1747 tokens, short): decode setup 97 ms, arena released + 25 s prefill OK + 3103 ms decode build, then 98 ms again; exit 0. WITHOUT the per-prompt policy the long prompt crashed (CUDA OOM, exit 134).
Opt-outs: `MOESTREAM_QWEN4EXP_DECODE_RECREATE_GRAPH_ARENA=1` (no borrowing), `MOESTREAM_QWEN4EXP_GRAPH_BOOTSTRAP=0` (no startup preparation). Possible follow-up: prefill scheduler activation copies (workspace ~7.5 MiB/token) would raise the prompt length that keeps the arena resident.

## 19. Host-node gap investigation: it was thread placement, not the signaling (2026-10-02, late)

Host-node mode (CPU MoE as CUDA host callbacks) ran at 22.7 tok/s vs 28.0 for plain 22/26 splits. Findings, in the order they were established:
- Per-call breakdown of the runner (`MOESTREAM_QWEN_CPU_GRAPH_TIMING_REPORT_CALLS`): input copy 0.9 us, metadata 0.08 us, output copy 1 us, **compute 1146 us**. The body is entirely the ggml graph compute, ~30% slower than the same layer as a plain scheduler split (~0.87 ms).
- Handoff overhead measured two ways is small: an `nsys` trace gives D2H-end -> H2D-start = 1555 us vs callback body 1495 us, i.e. ~60 us per layer; a standalone CUDA benchmark (`handoff.cu`-style: D2H + `cudaLaunchHostFunc` + H2D in a captured graph) gives 29 us per handoff at the real ~10 KB payload.
- Signal/doorbell alternative (GPU kernel posts a flag in mapped host memory, persistent CPU thread spins, GPU wait-kernel spins on the reply): 4.6 us (256 B), 12.8 us (10 KB), 38 us (40 KB) per handoff vs 23.5 / 29.3 / 35.1 us for the host callback. At the real payload that saves ~16 us/layer, ~0.4 ms/token over 26 layers, so it is NOT the main lever.
- CPU weighted-reduce tail on the CPU (mul + 10 views + 9 adds + cont, 25 nodes vs 4): only ~37 us/layer (barrier ~1.8 us/node). The runner's `output_per_expert` mode (GPU does the reduce) could save ~1 ms/token.
- Root cause, reproduced standalone (`cbthread*.cpp`): the SAME ggml graph computes 727-750 us from the main thread but 975-993 us from a CUDA host-callback thread (+33%). Not affinity masks (12 CPUs for both), not thread churn (one stable callback tid), not a second OpenMP pool. It is worker placement: binding the OpenMP team (`OMP_PLACES=threads OMP_PROC_BIND=close`) gives 787 us; ggml's own strict CPU mask gives 782 us. `OMP_WAIT_POLICY=active` with the callback design is dangerous (6945 us: spinning workers + a busy main thread leave no CPU for the callback thread).
- End-to-end on host-node mode (3 interleaved reps each, idle machine): env binding `threads/close` 25.37 (+11.5%), default policy + bind 25.30, `cores/spread` 21.73 (-4.5%), spin vs default policy irrelevant.
- Binding does NOT help the plain modes: 22/26 +4.4% (noise-level), 0/44/4 -4.2% (two of three reps -8%). So it is applied only in host-node mode.
- Implemented in `src/ggml_qwen4exp_graph.cpp` (`pin_host_node_threads`): in host-node mode the DECODE CPU thread pool (the one the callbacks borrow) uses ggml `strict_cpu` with a mask of the first N allowed CPUs, and the main thread's affinity is restored after pool creation. Pinning the prefill pool too (first attempt) collapsed decode to 4.76 tok/s: ggml pins the creating thread and its workers permanently, so two teams ended up pinned to the same CPUs.
  Result (2 interleaved reps): in-engine pin **26.25 tok/s (+15.6%)** vs 22.70 opt-out vs 25.57 env binding; output identical. `MOESTREAM_QWEN4EXP_HOST_NODE_PIN=0` disables it. Host-node is now within ~1.75 tok/s of plain splits.
- Answer to "wait for all CPU work, then signal once": already the case within a layer (one callback runs gate+up+swiglu+down+reduce, then the H2D copy releases the GPU). Across layers it cannot be batched: layer L+1's GPU attention/router needs layer L's MoE output.
- Remaining gap (~2.4 ms/token): handoff ~1.6 ms (doorbell -0.4, packing 2.5 D2H copies into one ~-0.4), CPU reduce tail ~0.9 ms (`output_per_expert`), residual placement effect ~1.3 ms. Not implemented.
