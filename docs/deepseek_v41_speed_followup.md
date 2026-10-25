# DeepSeek V4.1 Flash: 10 tok/s decode and 100 tok/s prefill follow-up

Active targets requested after the 5.71 tok/s checkpoint: use both the RTX
3090 and RTX 5060 Ti, reach about 10 tok/s decode and 100 tok/s prefill.
Neither new target has been verified yet. Preserve the single native
DDR/VRAM expert cache; no graph-owned expert arena or duplicate weight pool.

## Reference reviewed

`/home/ray/work/moestream-graph-framework`, branch `graph-framework`, head
`ff96893`. See `docs/ssd-offload/13-glm-decode-results-2026-10-14.md` there.
The GLM 805-token prefill improvements are GPU batched MoE, transient-expert
read-ahead, and computing GPU1-hot experts in place. Commit `4570144`
reports 111.7 -> 159.2 tok/s on two GPUs (117.8 on one 3090). The later
speculative device ring is opt-in and only about 3.5% faster; it is not the
main improvement to copy. GLM's decode results use last-30-token timing;
compare that separately from complete-request decode timing.

## Implemented and initial evidence

- Extended native grouped MoE prefill to DeepSeek: gather by expert, use live
  VRAM cache pointers before considering host loads, retain compute holds
  through chunk completion, and download separate `(token,k)` activation
  rows for the same ordered FMA reduction as the validated decoder. No
  expert weights enter a graph arena. This is opt-in through
  `moe_prefill_batched=true`; the shipped profile is unchanged for now.
- Grouping alone was not a speed win. At 19 / 86 input tokens the preserved
  stable binary measured 19.74 / 53.08 seconds, while the initial grouped
  candidate measured 22.27 / 56.22 seconds. All 16 subsequent logit hashes
  matched exactly, so the batching preserves these cases' math.
- The 86-token baseline spent 28.54 seconds in attention output projection.
  Extended the exact Q2_K grouped-output CUDA kernel from one token to a
  complete prompt batch. Both GPUs passed all 51,444 checked output values
  bitwise against the actual AVX-512 dot, including 3- and 17-token batches,
  padding and output guards. Weight bytes were unchanged.
- With that GPU output projection and asynchronous RAM read-ahead, prefill
  improved to 7.39 seconds / 2.57 tok/s at 19 tokens and 11.73 seconds /
  7.33 tok/s at 86 tokens. All 16 full logit hashes still matched the stable
  reference exactly. This is progress, not evidence for the 100 tok/s target.
- Removed the compressed-attention host F32 weight mirror on successful GPU
  projection; the native fallback initializes it lazily. The compiled engine
  now includes this memory reduction; the 19/86-token measurement above was
  before that last host-only optimization.
- Grouped read-ahead follows expert execution order and skips unused
  active-area warmup copies. It uses the existing prefetch workers and RAM
  claims. The candidate enables `prefill_warm_via_prefetcher` and the bounded
  prefill window capacity, matching the GLM structural approach.
- Found and fixed another memory reuse issue: synchronous MOVE promotion
  cleared `slab_capacity` without freeing its buffer. The next same-size load
  could unnecessarily free/reallocate it. Keep the real allocation capacity.
  The focused test of real RAM promotion/reuse bookkeeping passed, including
  COPY and a pinned MOVE case.

## Full DDR results

The full 100 GiB DDR candidate (`pin_ram_fraction=1.0`,
`move_on_promotion=false`) first finished three 64-output prompts at **6.56 /
7.69 / 5.31 tok/s**, combined **6.37 tok/s** (192 tokens / 30.13 s). The DDR
tier filled all 8,360 slots (107.094 decimal GB). With the same DDR cache fully
page-locked (`ram_pinned_budget_gb=107.1`), the same prompts measured **7.62 /
8.63 / 6.13 tok/s**, combined **7.31 tok/s** (192 tokens / 26.28 s). All 192
logit hashes still match the prior reference. Decode is faster but remains below
the 10 tok/s target.

The first 817-token prefill experiment measured **6.14 cold / 6.28 warm tok/s**.
Its log exposed that only 128 tokens used batched prefill; the remaining 689
were processed as single-token forward calls. The DeepSeek adapter now defaults
to the whole prompt (`ZEROMOE_DEEPSEEK_V41_PREFILL_CHUNK=0`). The old indexed
attention limit also no longer applies to the native attention callback.

## GPU graph migration (experimental, not validated for adoption)

The GLM design requires GPU graph nodes for the numerical trunk, rather than
wrapping the native host forward in CPU graph callbacks. New opt-ins:

- `ZEROMOE_DEEPSEEK_V41_GPU_PREFILL=1`: GPU Engram projections, HC,
  compressed attention and dense projections. The reference callback path
  remains available with this unset. GPU graph build failure stops the request.
- `ZEROMOE_DEEPSEEK_V41_CACHE_MOE_GRAPH=1`: gate/up/SwiGLU/down and branch
  weighting/sums run as GPU graph nodes on both GPUs. Bind existing native
  VRAM pools as strided weight tensors; allocate **zero graph expert bytes**.
  Startup selects the existing uniform native cache layout, using the model's
  row-compatible 12,862,080-byte slot stride. DDR ownership is unchanged.
- Cache callbacks resolve required expert identities, load only real cache
  misses, and hold every device source through completion. Masked routes use
  distinct, valid held slots. The next layer's dependent callback retires the
  preceding layer's holds; final holds retire after graph synchronization.
- Raw SWA, compressed/indexer KV, partial compressor blocks and final HC mix
  are explicitly transferred back to existing native state for decode.
- Corrected tensor layouts in the previously unused GPU builders: ratio-1/2
  pooling, Engram stream strides, MQA head broadcasting, and indexer key
  sharing by the KV owner. Long prompts include GPU indexer top-k masks.

The 19-token GPU trunk and dual-GPU cache graph completed smoke runs, but
whole-model outputs did not yet match the native reference. At default CUDA
TF32 settings, cache-graph versus GPU-trunk logit relative RMS was about
0.34–0.44. An F32 diagnostic mode with TF32 disabled reduced layer-zero
attention/HC/MoE trace errors to roughly 1e-6, but the first router selection
still changed at layer 8 after accumulated drift. This remains a correctness
failure; the F32 mode is diagnostic and has not been adopted.

The Q8 activation conversion was a large contributor: layer-zero Q-lora RMS
error fell from 5.7e-3 to 1.5e-7 when replacing it with native Q2_K F32
matmuls. GGML's cuBLAS handle also enables TF32, so comparisons of F32
operations must set `NVIDIA_TF32_OVERRIDE=0`.

A zero-copy cache-view regression previously compared the strided expert
weights with planar GGML weights on **both GPUs**: all **438,784 F32 outputs
were bitwise identical**, including 1/19/817-token batches and the real
5120-by-2304 geometry. It also checked that weight bytes and slot padding were
unchanged. The optional native F32 hook subsequently passed direct
bitwise comparisons against the original Q2_K/Q3_K projection kernels on
both GPUs, including negative masked routes and grouped output tensor strides.
Those checks establish the arithmetic and cache-view contract, not whole-model
correctness.

`bench/build_dsv41_f32_backend.py` builds the optional F32 graph hook in an
isolated output directory from the recent GLM GGML object files, and applies
[the local hook patch](../patches/ggml/0010-native-f32-quant-graph-hook.patch).
It does not change either shared GGML source tree. The opt-in
`ZEROMOE_DEEPSEEK_V41_NATIVE_F32_GRAPH=1` uses the original quantized dense
and expert weights, with F32 activations, and allocates no graph expert pool.
A model run requires both this patched backend and
`NVIDIA_TF32_OVERRIDE=0`; build or runtime failure stops the request instead of
falling back silently. The first paired model run built the GPU graph but its
10 GiB MemAvailable check rejected compute at 9.7 GiB. That graph-only check
has since been unified with the host allocation floor. The native reference
from that pair completed. The rerun completed the 19-token GPU
graph in 14.52 s (1.31 tok/s), compared with the saved native reference at
2.20 tok/s. All eight teacher-forced argmax tokens agreed, but none of the
full logit hashes did; relative RMS error ranged from 0.187 to 0.316. This
is still a model-level correctness failure.

Per-layer graph timing attributed 6.34 s to routing/warmup and 7.42 s to
expert resolution/upload across 40 layers. The graph reported 733 synchronous
main-cache misses and the process reported 45.85 GB of GPU uploads, indicating
substantial extra cache traffic. The graph callback no longer invokes native
prefill warmup, which fills the separate active-layer pool and triggers
speculative promotions that its graph nodes never consume. It still stamps
actual prefill demand for decode retention. This change compiled; its speed
on a second 19-token run improved to 6.89 s (2.76 tok/s). A full 40-layer
native/graph activation trace places the first large error at layer 8 MoE:
the last prompt token has 0.295 relative RMS error there, while the first
18 tokens and preceding layers remain near the native path. A focused route
trace found five shared experts and one swap (native ID 336, graph ID 311).
The graph therefore still fails the model-level correctness gate.
The paired score capture confirmed the native selection score for ID 336
exceeds ID 311 by only 3.53e-5; the graph reverses that by 3.81e-6. The
earlier layers' activation drift crosses this top-k boundary. Changing cache
or expert weight layout will not fix the swap; the graph's F32 attention/HC
arithmetic needs closer agreement with the native path, or a separately
validated policy for such near-ties. A roughly 800-token graph prefill run is
queued to measure the long-batch speed separately.

The 817-token full-graph run did not complete. Its durable log stops at layer
19, so it provides no prefill throughput result. The prior 6.14 / 6.28 tok/s
result used a 128-token GPU batch plus 689 single-token steps and does not
measure the full GPU trunk.

A separate GLM benchmark queue has repeatedly occupied the GPUs. Device
queries require host access; do not infer a driver failure from a sandbox
query. Do not interrupt the other queue. The guarded runner waits for real
compute processes to disappear and sufficient MemAvailable.

## Reproduction

Temporary `/tmp` logs can be cleared between runs. Recreate the engine and
focused targets with:

```bash
cmake -S . -B <build-dir> \
  -DZEROMOE_USE_GGML=ON -DZEROMOE_USE_GGML_CUDA=ON \
  -DZEROMOE_ENABLE_CUDA_GRAPHS=ON \
  -DZEROMOE_GGML_ROOT=/home/ray/Projects/llama.cpp-moestream-b11401/ggml \
  -DZEROMOE_GGML_PREBUILT_DIR=/home/ray/Projects/ZeroMoE/build_b11401/ggml-build \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build <build-dir> --target zeromoe zeromoe_tests \
  dsv41_cache_graph_test dsv41_grouped_outproj_test -j4
```

Build the optional native F32 backend from the latest GLM build, then validate
the original projection kernels and the two GPU cache views:

```bash
python3 bench/build_dsv41_f32_backend.py \
  --ggml-source /home/ray/work/ggml-iq2xxs/ggml \
  --ggml-build /home/ray/work/moestream-graph-framework/build_native/ggml-build \
  --out <isolated-backend-dir>
CUDA_VISIBLE_DEVICES=0,1 NVIDIA_TF32_OVERRIDE=0 \
  LD_LIBRARY_PATH=<isolated-backend-dir> ZEROMOE_TEST_NATIVE_F32=1 \
  <build-dir>/dsv41_cache_graph_test
```

`make test BUILD_DIR=<relative-build-dir> JOBS=4` reaches the EXL3 CUDA test and
fails in the upstream `cudaErrorNotSupported` stub because the ExLlamaV3 source
is not configured. The preceding EXL3 CPU, DDR cache and graph-config checks
pass; the complete native suite has not passed.

For the long prompt, `bench/dsv41_graph_profile.py` waits for both GPUs and
host RAM to be idle, preserves the selected DDR and pin settings, and sends
one input line (43 repetitions, about 818 tokens). Its `--output` directory
preserves the exact prompt, effective config, full logs, live-memory samples,
and timing summary. It terminates the model if MemAvailable falls below 26 GiB.
Run it after building the isolated backend above:

```bash
python3 bench/dsv41_graph_profile.py \
  --binary /tmp/zeromoe-v41-build/zeromoe \
  --backend /tmp/v41-ggml-f32-libs \
  --config configs/bench/deepseek_v41_100gib_ddr_17gib_vram_dual_gpu.toml \
  --output logs/perf/deepseek_v41_graph_818_20261015
```

No GGUF was modified, repacked, replaced or deleted in this follow-up.

## Host hang investigation and guard fix

The durable 817-token run log shows the benchmark driver had changed the
configured `ram_pinned_budget_gb=15` to **107.1 GiB** and `pin_ram_fraction=0.5`
to 1.0. Startup had only **129.1 GB** available against **126.3 GB** required
by the 115.6 GB host budget plus the old 10.7 GB margin. It then allocated
**107.094 GB** of expert slabs, with 3.23 GB resident tensors, and requested
a **107.1 GB** page-pin ceiling that matched the entire slab pool. The graph
workspace was 1.11 GiB. The run ended before logging the final pinned-byte
count. Its last recorded layer was 19 at 22:17; the host booted again at
22:24. This makes near-total page-locked DDR pressure the leading cause,
though the surviving logs cannot prove the exact kernel failure. The driver
no longer rewrites the pin settings.

The same log has six `ggml-cpu/ops.cpp:563: not implemented` messages. That
line calls `abort()` when the CPU backend tries to duplicate an I32 tensor to
a type other than F32. This is a confirmed graph execution failure and likely
terminated the engine process; it is separate from the host reboot and needs
a graph placement/type fix before another prefill run.

Fixed the indexer mask path by keeping `ggml_top_k`'s I32 row indices. The
following `ggml_set_rows` accepts I32 directly, so the unnecessary I32-to-I64
`GGML_OP_CPY` is gone. A guarded 817-token verification completed graph
compute with status 0 across all 40 layers, emitted no `ops.cpp:563` message,
and reached `PREFILL-DONE` at **12.88 tok/s**. The monitor recorded a minimum
MemAvailable of **42.39 GiB** under a 90 GB host cap, the 15 GiB pin limit,
and a 26 GiB emergency stop. This confirmed the CPU fallback fix.

The layer-8 near-tie was resolved in the GPU-prefill router only: if the kth
score is within **2e-5** of the next score, order that boundary by descending
expert ID. The original layer-8 pair had a graph gap of 3.81e-6 and native gap
3.53e-5; both now select expert 336. A paired 19-token trace shows identical
route IDs through layer 8 and reduces the last-token layer-8 MoE relative RMS
from 0.295 to 5.82e-5. A wider 1e-4 margin incorrectly changed a correctly
ordered layer-10 boundary (native gap 7.34e-5), so it was narrowed and covered
by a component test. Other route sets begin to diverge again at layer 19; full
model numerical parity is still unproven, and prefill remains far below the
100 tok/s target.

The prior boot's kernel journal was not persistent, so there is no surviving
OOM-killer, NVIDIA Xid, or PCIe record to establish the exact hardware failure.
The current kernel was booted with `nowatchdog`, so it could not report a
soft-lockup diagnosis. The `ggml-cpu/ops.cpp:563: not implemented` messages in
the run log are a separate graph fallback issue and also need investigation.

Host admission now counts outstanding reservations against the startup
MemAvailable snapshot. This closes the lazy-allocation race where parallel
`posix_memalign` calls reserve virtual DDR before pages become visible in the
kernel's live MemAvailable count. Every new reservation also checks current
MemAvailable, failing closed if `/proc/meminfo` cannot be read. Startup and
runtime guards now preserve at least **20 GiB** for Linux and drivers. The
dual-GPU 100 GB profile is capped at a 107 GB total host budget, keeps the
configured 15 GiB page-pin limit, and still requires current headroom before
starting. The guarded 19-token smoke subsequently completed all 40 graph
layers at a temporary 100 GB host cap. Its monitor recorded a minimum
MemAvailable of **57.35 GiB**, above the 26 GiB emergency floor; it reported
`emergency_stop=false`, zero runtime errors, and graph compute status 0. The
effective config retained `ram_pinned_budget_gb=15`,
`pin_ram_fraction=0.50`, and `move_on_promotion=true`. This validates the
guarded short run only; it does not validate long-prompt stability or speed.

## Post-merge decode profile (2026-10-16)

The latest merged source was built with the GLM GGML libraries and profiled
with native prefill, persistent full-graph decode, both GPUs, a **90 GB host
cap**, the profile's **15 GiB page-pin limit**, and a **26 GiB emergency
floor**. A 19-token prompt followed by 64 generated tokens completed with no
emergency stop; the lowest sampled MemAvailable was **63.7 GiB**. Decode wall
time was **12.342 s / 5.19 tok/s**. This single-prompt, 90 GB result is not
directly comparable with the earlier three-prompt, fully pinned 100 GiB DDR
result of 7.31 tok/s.

New decode-only timers in `ggml_deepseek_v41_graph_decode.cpp` attribute the
62 warm graph runs (192.4 ms each) as follows:

| Stage | Mean per token |
| --- | ---: |
| Cache-aware MoE callback | 140.4 ms |
| Attention host callbacks | 26.7 ms |
| Native Q/K/V projection | 11.3 ms |
| HC and Engram callbacks | 9.4 ms |
| Router lookahead | 3.4 ms |
| Logits head | 1.0 ms |

The MoE callback alone exceeds the 100 ms/token budget for 10 tok/s. The run
reported 373 real decode-time SSD fetches, and the end-of-request caches held
2,555 VRAM and 3,611 DDR expert slots. The next decode change should target
MoE miss resolution and transfer overlap; removing router lookahead alone
cannot close the gap. The guarded log is under
`/tmp/v41-decode-profile-20261016/` while temporary files remain available.

A repeat guarded run measured **5.38 tok/s**, with 134.5 ms/token in MoE.
Decode-only counter deltas show **57.1 ms/token** in batched RAM expert
resolution, **45.9 ms/token** in active-layer uploads, and **39.5 ms/token**
of disk reads. These counters overlap the enclosing MoE callback and each
other; they must not be added together. The decode phase read **4.765 GB**
and uploaded **5.419 GB** of expert data. The active-layer copy report shows
1,663 synchronous uploads from pageable slabs over the complete request,
versus 506 from pinned slabs. The graph's depth-1 pilot issued disk hints but
made **zero** speculative RAM-to-VRAM promotions. The repeat log is under
`/tmp/v41-decode-moe-breakdown-20261016/`.

Controlled 90 GB A/B runs with the existing decode async prewarm switch were
slower than the unchanged configuration: blocking prewarm **5.21 tok/s** and
nonblocking prewarm **5.12 tok/s**, compared with **5.38 tok/s** in the
immediately preceding baseline. All three completed without an emergency
stop. The shipped `decode_async_prewarm=false` setting remains appropriate
for this workload.

The cache graph's first compact-layout attempt used an expert-plane stride
that was not a whole number of GGML matrix rows. The layer-0 route IDs still
matched native decode, but its MoE output contained **5,054 NaNs** and the
following logits diverged. The unchecked view is now rejected unless each
layer's slot region passes bounds, contiguity, size, and row-stride checks.

For the opt-in graph path, startup now rounds each compact layer's slot size
to the alignment returned by `expert_cache_layer_stride_alignment`. This
keeps slots in the existing GpuTopology pool and preserves its ownership;
the cache slot count is recalculated from the padded sizes. A guarded
64-token run with this layout completed all 40 layers, had **58.37 GiB**
minimum MemAvailable and no emergency stop. Its decoded stdout matched the
native path byte for byte for this prompt. Decode was **5.21 tok/s** overall
with **191.9 ms** warm graph compute per token, so the graph is correct on
this sequence but does not improve speed yet. Full-logit parity remains
unverified; next work needs to cut the GPU MoE and scheduler time.

## GLM-style cold-expert CPU split (2026-10-16)

The non-cache-graph DeepSeek decode callback used the ordinary MoE dispatcher.
With the current 100 GB host cap and 15 GiB pin limit it sent **19,680/19,680
expert calls to GPU** and zero to CPU. Its decode rate was **5.76 tok/s** for
64 generated tokens; decode-only counters reported 47.4 ms/token in batched
RAM resolution, 45.2 ms/token in active-layer uploads, and 34.0 ms/token in
disk reads. This is the dispatcher behavior that differed from GLM's explicit
hot-GPU/cold-CPU lane.

The existing `cpu_offload_max_experts` cap now has a DeepSeek decode policy:
when enabled, it can send DDR-resident experts outside both persistent and
active-layer VRAM caches to the CPU. Main-cache and active-layer hits stay on
GPU. The policy is restricted to single-token decode so prefill retains its
existing path. It remains opt-in; the shipped DeepSeek config still leaves the
cap at zero.

A guarded 64-token run with the cap set to eight completed without emergency
stop, matched the native output exactly, and split **17,613 GPU / 2,067 CPU
expert calls**. Active-layer uploads fell to zero, but decode was **5.73
tok/s**, within noise of the all-GPU 5.76 tok/s baseline. The hybrid policy
alone does not meet the 10 tok/s target; it needs to run cold CPU work in
parallel with a batched GPU hot-expert branch, following GLM's early-ready
lane. Minimum MemAvailable was **58.33 GiB**.

The compact cache-graph callback's `upload_ms` is an inclusive service timer
from route completion to payload binding. It includes DDR expert lookup,
possible SSD loads, VRAM upload work, and cache bookkeeping; it is not a
PCIe-only measurement. In the async 64-token graph run, the callback recorded
1,206 routed VRAM misses (about 19 per generated token). Multiplying by the
12.862 MB slot size gives **15.5 GB of possible payload**, not measured H2D
bytes. Whole-process H2D counters include several upload paths and cannot be
used as the graph callback's transfer total. A separate native-callback run
measured **4.612 GB** of decode disk reads and **5.444 GB** of decode GPU
uploads; those are from that run, not a direct decomposition of the graph
callback timer.

The guarded async cache-graph upload retry also completed all 40 layers with
no emergency stop and byte-identical output to the native run. It measured
**5.47 tok/s** versus **5.36 tok/s** in the synchronous graph run; these are
single runs and show no material improvement yet. The async path remains
opt-in.

Batching the cache graph's DDR lookups with `get_experts_batch` improved the
same guarded 64-token profile to **5.71 tok/s**; decoded stdout remained
byte-identical to native and minimum MemAvailable was **53.42 GiB**. Its
inclusive route-to-payload service timer fell from **6.52 s to 5.84 s** over
the 63 decode steps. These are single runs; the target remains unmet.

With the verified async slot-hold path enabled on top of batched DDR lookup,
the same profile measured **5.76 tok/s**, **5.64 s** inclusive route-to-payload
time, and **53.33 GiB** minimum MemAvailable. It matched native stdout and
reported no emergency stop. This remains a single-run result.

The first graph-level cold CPU lane (`ZEROMOE_DEEPSEEK_V41_CACHE_MOE_CPU_COLD=1`)
starts a batched DDR lookup and per-expert CPU FFNs while the graph computes
main-VRAM hits, then merges the CPU partial output after the GPU branch. Its
guarded 64-token run completed all 40 layers, matched native stdout, had no
emergency stop, and kept **53.54 GiB** minimum MemAvailable. It recorded
**2,129 CPU / 4,560 GPU FFN calls** and measured **4.57 tok/s**, slower than
the GPU-only cache graph's 5.47 tok/s. This scalar CPU path is a correctness
prototype; the next iteration needs GLM's batched GGML CPU expert graph to
reduce cold-branch compute time. A timed repeat saw 1,373 cold-lane callbacks,
8.90 s total worker time, and 7.47 s waiting at the merge, so most cold work
was still on the token's critical path rather than hidden behind GPU compute.

The first cache-graph CPU-cold branch used an asynchronous host task for DDR
loads and per-expert CPU FFNs, while the graph computed main-VRAM hits and
merged both results afterward. A guarded 64-token run completed all 40 layers,
matched native stdout, preserved **53.54 GiB** minimum MemAvailable, and had no
emergency stop. It routed **2,129** experts to CPU and **4,560** GPU FFN calls,
but decode fell to **4.57 tok/s**. The per-expert CPU path is too slow; the next
implementation needs GLM's batched CPU graph execution and earlier scheduling.
