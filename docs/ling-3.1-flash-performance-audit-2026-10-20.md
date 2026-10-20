# Ling performance audit against the tuned GLM and DeepSeek worktrees

The requested 10 tok/s decode and 20 tok/s prefill are **not yet delivered**.
The earlier 6–7 tok/s Ling numbers were graph execution rates; the measured
32-token request was about 1.3 tok/s because it also built two graphs. A
profile fitted to one reply reaches 10.16 tok/s on that forced reply, but a
free-running check falls to 5.22 tok/s. That conditional result is not a
general decode speed claim.

## References inspected

All peer worktrees were read without modification.

| Reference | Worktree / evidence |
| --- | --- |
| GLM 5.3, non-Flash | `/home/ray/work/moestream-glm53`, committed HEAD `4e5c8ce`; `docs/GLM5.3-baseline.md`, `bench/glm53/`, and the committed implementation. The working tree has an unfinished merge, so its conflicted files are not treated as a runnable baseline. |
| GLM 5.3 Flash | `/home/ray/work/moestream-graph-framework`, HEAD `7344dfa`; `bench/glm_recipe_env.sh`, `docs/ssd-offload/11-glm53flash-handoff-2026-10-11.md`, and `13-glm-decode-results-2026-10-14.md`. Also checked the older `/home/ray/work/moestream-graph` tree. |
| DeepSeek V4.1 Flash | `/home/ray/work/moestream-deepseek-v41-flash`, HEAD `a03a879`, and `/home/ray/work/moestream-dsv41-on-master`, HEAD `0737a17`; configs, graph implementation, decode report and speed follow-up. |
| Ling | `/home/ray/Projects/MoEStream/.worktrees/ling-3.1-flash`, base `10cc7ec`, plus the in-progress changes. |

The GLM Flash report measures the last 30 decoded steps on identical
teacher-forced tokens. Its typical sustained result is 13–15 tok/s, with
faster windows. The non-Flash GLM report measures 2.00–2.09 tok/s
first-output-to-last-output, and roughly 13 tok/s prefill on a 426-token
prompt. These denominators differ from Ling's shutdown request rate.
The checked-in DeepSeek reports retain 5.71 and 7.31 tok/s checkpoints;
they do not establish the user's later 10–20 tok/s result or its recipe.

## Findings

### 1. The tested Ling placement does not keep 95% of its experts resident

| Item | Ling test | GLM Flash reference |
| --- | ---: | ---: |
| Actual configured GGUF files | 141.81 GB, four AD-IQ1_S shards | 87.35 GB, two AJ-IQ2_XXS shards |
| Layers / routed MoE layers | 72 / 68 | 45 / 42 |
| Experts per layer / top-k | 512 / 8 | 288 / 8 |
| Hidden / expert intermediate | 4096 / 1280 | 4096 / 2304 |
| Routed expert weight footprint | about 114.92 GiB | 74.21 GiB |
| Graph dense arena at decode | about 16.5 GiB | about 6.5 GiB |
| Hot expert budgets | 3 GiB on 3090 + 12.5 GiB on 5060 Ti | about 28.9 GiB across both cards |
| DDR expert pool | 72 GiB | auto, about 76 GiB capacity |

Ling's hot sets and DDR pool cover about 87.5/114.92 = 76% of expert bytes,
before considering which experts are actually selected. Native host dense
weights also remain resident (about 17.4 GiB), in addition to device dense
weights. Increasing the DDR *expert pool* to 100 GiB therefore does not mean
100 GiB total host use. The 72 GiB trace run peaked at 99.53 GiB cgroup use
and had 23.99 GiB minimum available RAM.

The DeepSeek reference has a much smaller native dense pool, about 3 GiB,
shared with graph nodes, and about 30 GiB of main expert caches across the
cards. Its memory ownership is a useful reference. Blindly copying its
expert-pool sizes into Ling would omit Ling's additional dense footprint.

### 2. First-request graph construction dominates the reported short run

The existing 72 GiB log constructs the prefill graph in approximately
4.6 seconds, executes the two-token prefill in 0.89 seconds, then constructs
the decode graph in approximately 14.3 seconds. Its 31 decode steps average
148.5 ms. Thus 6.74 tok/s per step and 1.31 tok/s for the request are both
descriptions of that run, with very different timing scopes.

The lifecycle keeps one `g_graph`: prefill and decode replace each other.
The next request's prefill also destroys the decode graph. This is not
necessarily just one startup cost for an entire interactive session.
The CPU lanes, assist graphs, and dense device allocations must be examined
for reuse; merely excluding this time from a report does not fix latency.

### 3. The simple usage file does not reproduce current full-graph routing

The old profile has 10,396 positive entries and 35,904 total counts. A new
32-token, free-running full-graph trace of the same raw prompt `Hi there`
contains 16,864 selections over 31 steps and 5,629 unique experts. Only
77.45% of those selections appear in the old profile.

The first 16 decode steps of the baseline report 336 expert loads, 64 ms
read time/token, 31 ms residual split stall/token, and 81 ms CPU-lane
compute/token. Read time and overlapped compute must not simply be added.

Ranking the new trace reduces CPU compute to about 40 ms/token and reaches
10.16 tok/s when consuming the original output token sequence. However,
moving experts between CPU and CUDA changes numerical results and the
greedy reply. The free reply uses a different set of experts and measures
5.22 tok/s. A useful simple profile needs to cover the observed free
replies, and each timing change needs both an identical-token comparison
and a free-running check. A full survey remains deferred.

### 4. A GGML API change breaks transfer of the GLM prefill recipe

The configured Ling build links prebuilt GGML libraries. Its newer
`ggml_backend_sched_copy_input()` calls an application weight-copy
callback, then falls back to whole-tensor copies when no callback handles
the input. Current llama.cpp registers
`llama_context::sched_copy_experts()` with
`ggml_backend_sched_set_copy_callback()`.

The older GLM GGML fork implements selected-expert copies inside the
scheduler, hardened by patches 0007 and 0008: skip negative routes, find
the consuming `MUL_MAT_ID` beyond the first node, copy each registered
expert separately, and supply padding without reading adjacent empty
source slots. Matching patch filenames in MoEStream do not prove those
edits applied to the newer, rewritten scheduler. Ling had no application
copy callback registered.

Also, `GGML_OP_OFFLOAD_MIN_BATCH` defaults to 32. Consequently setting
`MOESTREAM_GLM5NEXT_PREFILL_OP_OFFLOAD=1` alone still leaves the two-token
expert matmuls on CPU. Lowering the threshold to 2 places them on CUDA
(GPU workspace 1.22 GiB versus 0.27 GiB), exposing the missing selective
copy behavior. The uncorrected diagnostic failed in `cudaMemcpyAsync`
with `invalid argument`, and available host RAM briefly fell to 11.39 GiB.
Its cgroup swap remained zero. The monitor sent a termination signal;
the process aborted and all resources were released. This is a failed
diagnostic, not a usable configuration or speed measurement.

The integration fix adds `backends/ggml/ggml_expert_copy.hpp` and registers
the newer API callback on the shared GLM/Ling schedulers. It retains the
legacy patched-scheduler path when that API is absent. Route selection,
weights and quantization are unchanged. The callback resets cached IDs
on every graph execution and reports actual transferred plane bytes.
The protected-page regression passed, followed by a full 72-layer
prefill/decode run using a 64 GiB DDR pool, 96 GiB kernel memory cap and
26 GiB monitor stop. It completed 2,022 plane copies / 2.231 GiB of actual
weight payload without the copy failure or memory surge. Minimum
available RAM was 31.32 GiB, peak cgroup use 89.24 GiB, and swap and OOM
events were zero. Only C++ was rebuilt, under a separate 16 GiB no-swap
build cap; CUDA kernels and the CUDA library were unchanged.

This two-token GPU prefill took 1.06 seconds (1.89 tok/s), so it is slower
than CPU prefill on this tiny batch and remains opt-in. The fixed-token
decode steps averaged 108.44 ms (9.22 tok/s); request-level decode was
1.43 tok/s including setup. The smaller pool also changes the miss rate,
so this run validates the transfer fix, not an isolated speed gain.
The old zero-valued core H2D counters did not count these GGML transfers;
they must not be interpreted as proof of zero H2D traffic.

### 5. Two useful GLM optimizations are missing from the actual Ling build

The current CPU library's IQ2_XXS dot uses its AVX2 path; the GLM reference
fork has the AVX-512BW/VL expansion at
`ggml_vec_dot_iq2_xxs_q8_K`. Ling's dominant 63 layers have IQ1_S gate/up
and IQ2_XXS down, so this can help the down projection. The IQ1_S function
is identical in the inspected sources; copying the IQ2 kernel does not
automatically accelerate IQ1_S gate/up. This is a CPU-library rebuild,
not a reason to rebuild CUDA kernels.

The latest non-Flash GLM commit also implements opt-in
`MOESTREAM_OFFLOAD_RETARGET=1` page-reusing eviction. Ling's current
`memory/offload_store.hpp` lacks it. The GLM controlled result improved
roughly 1.75 to 2.05 tok/s by avoiding page discard/refault costs. It is
relevant to Ling's measured misses, but must be ported and measured,
including mixed quant strides and registered-page ownership.

Ling **does use CPU offload**: 1,088 private CPU-lane calls in the first
16 decode steps. The problem is not a zero-CPU configuration in these
latest runs. Its pinned fraction is 0.2 versus the GLM recipe's 0.5;
only resident, pinned experts are eligible for GPU assist. Thread count
and pinning need an isolated comparison, rather than assuming GLM's
IQ2-optimal settings are IQ1-optimal as well.

## Evidence and next work

`bench/ling/reference-audit/` preserves the exact prompt, original profile,
trace-derived profile, original output token sequence, TOML, and a JSON
manifest of five diagnostic logs and their SHA-256 values. Temporary
logs and 500 ms memory samples remain under `/tmp/ling-reference-*`.

Priority order:

1. Verify selected-expert copies on the current GGML API, first with
   protected unused source pages, then with a smaller DDR pool and tighter
   kernel memory cap. Keep the 20 GiB available-memory floor and zero swap.
2. Retest GPU prefill with bounded transfers. Measure short-prompt latency
   and a batched prompt separately; GLM's 805-token prefill result does not
   predict two-token throughput.
3. Address CPU/GPU placement feedback in the simple usage profile. Preserve
   free-running performance as a required check beside forced comparisons.
4. Bring over the measured CPU IQ2 kernel and page reuse separately, and
   test CPU thread count / pinning using identical tokens.
5. Remove repeated prompt/decode setup costs through resource reuse,
   without adding duplicate DDR or VRAM weight pools.

The target remains the original 10 tok/s decode and 20 tok/s prefill.
No completion or successful merge is claimed by this audit.
