# DeepSeek V4.1 Flash 1017 Engram onboarding

Status: **in progress**. The SSD-sidecar loader and same-layer dummy-route fix are in the current dirty master worktree. A guarded full-graph smoke completed prefill and eight decode tokens without NaNs. Full-fit, reference parity, cleanup, and performance gates remain open.

## Model package

Main shards and Engram table sidecar are in:

`/mnt/usb-ssd/@/home/ray/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/`

- Main: `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf` plus shards 2–5.
- Sidecar: `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`.
- Main architecture: `deepseek41`; 40 layers, hidden size 5120, 384 routed experts, top-k 6, one shared expert, HC multiplier 4.
- Main GGUF carries PLE/hash metadata for Engram layers 1 and 14. The sidecar architecture is `deepseek41-engram` and contains two IQ4_NL tables: `[256,384006168]` and `[256,384016682]`.
- Small main-shard projections use the packed names `engram_gate.weight` and `engram_kv.weight`.

The sidecar stays separate so its architecture metadata cannot override `deepseek41`. The runtime gathers table rows on demand from SSD with bounded row caching. No large GGUF was copied or repacked.

## Reference operation

Use the first-party [DeepSeek V4.1 model implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/model.py) and [Engram implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/refs%2Fpr%2F55/inference/engram.py). For each Engram layer, lookup the hashed rows, apply the WKV projection, split keys/value, apply the fused Q×K gate projection, normalize per HC stream, compute signed-square-root/sigmoid gating, then add the gated shared value to the residual. The local llama.cpp snapshot does not implement this V4.1 Engram variant, so teacher-forced boundary parity still needs a dedicated reference harness.

## Latest validation

Current repo is `/home/ray/Projects/ZeroMoE`, branch `master`, HEAD `e2167121bef0748954a435e5a8d3fad8ccbb4969`; the SSD Engram work remains uncommitted. Current clean binary `build_dsv41_engram_20261102/zeromoe` passes provenance: GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`, CUDA toolkit 13.3, ZeroMoE native targets sm86/sm120. The prebuilt GGML CUDA library is sm86-only, so matched performance is provisional until rebuilt for sm120.

A 97-token dual-GPU guarded smoke used an 85 GB host budget, 15 GiB pinned-memory limit, and 26 GiB MemAvailable emergency floor. It completed all 40 full-graph prefill layers and generated eight tokens with no NaN/Inf reports. Minimum MemAvailable was 71.34 GiB; no emergency stop. Artifact directory:

`build_mtp_results/2026-11-03-e216712-e216712-deepseek41-engram-same-layer-clean-smoke/`

Measured smoke rates were 0.74 tok/s prefill and 0.71 tok/s decode (0.63 tok/s decode wall), far below the active targets of 100 tok/s prefill and 10 tok/s decode. The process still exits -6 during CUDA `cudaFree` cleanup. `check_full_graph_moe_run.py --full-fit` fails because the adapter reports `memory_plan=Unknown`. The smoke is not onboarding acceptance.

## Layer-7 NaN root cause and fix

The cached graph created zero-weight route entries for experts owned by the other GPU. It chose dummy expert slots from any resident layer in the noncompact global cache. The current layer's quantized down projection then interpreted another layer's bytes; NaN outputs survived multiplication by zero. With pool order corrected, all selected layer-7 routes were finite and every non-finite down result belonged to a zero-weight dummy route. A same-input A/B had bitwise-identical layer input and routes: cached graph output contained 332,037 NaNs / 496,640 values, while the host callback contained none. Sampled selected VRAM expert slabs matched their DDR sources byte-for-byte.

The fix only chooses dummy slots from the same layer and reuses a same-layer slot when fewer than top-k distinct slots exist. A guarded run with the negative-ID experiment disabled completed prefill and decode without NaNs. The code change is present in the current worktree but has not been committed.

The NaN also reproduced on detached pre-settings-migration commit `b8b47a4` using its original `ZEROMOE_DEEPSEEK_V41_*` variables. That earlier test backported only the sidecar loader/PLE/fused Engram support and reduced the 5060 Ti MoE cache from 10 to 8.5 GB to preserve its 2 GiB VRAM floor. This rules out the typed environment-setting migration as the cause. Historical artifact: `build_mtp_results/2026-11-03-b8b47a4-deepseek41-engram-legacy-env-layer7-cache85/`.

## Remaining gates

1. Build a teacher-forced boundary comparison against the first-party V4.1 Engram sequence, including hidden states and routed IDs/weights.
2. Implement a defensible cache-backed decode memory plan; required full-fit checker currently fails on `Unknown`.
3. Resolve CUDA backend teardown (`cudaFree` / `driver shutting down`).
4. Profile SSD read and expert-cache costs; meet 100 tok/s prefill and 10 tok/s decode on matched repeatable workloads.
5. Rebuild GGML for both GPUs and run the full-graph correctness and matched llama.cpp performance gates.


## Why the earlier rate was higher
The pre-fix ~4–5 tok/s decode and 11–28 tok/s prefill measurements had NaNs after layer 7; later routing collapsed to only six unique experts per layer. That run read 8.61 GB of experts and uploaded 8.87 GB. With finite routing after the same-layer dummy fix, the 97-token prompt selected 4,271 expert blocks (36.87 GB) and moved 42 GB H2D. The clean run and immediate repeat both measured about 0.74 tok/s prefill / 0.71–0.76 tok/s decode, with 161 slow SSD reads around 200–250 ms and about 0.03 GB/s real read bandwidth. The high earlier rates are invalid; correct routing exposed SSD expert reads as the performance bottleneck. Async H2D overlap remains untested because the guard detected another ZeroMoE process and MemAvailable below the start threshold.


## Cache capacity and throughput drop
The full MoE weight set is 15,360 expert slots / 157.13 GB according to `gguf_tool plan` (Engram tables excluded). The smoke config used an 85 GB host cap, a 6,000-slot RAM-cache cap, and 21 GB total MoE VRAM cache; full residency cannot fit even before dense/graph overhead. `ram_preload_all_experts` was not enabled. The cache loads routed experts on demand: the corrected 97-token prompt loaded 4,271 unique expert blocks / 36.87 GB from SSD, with zero RAM-cache hits and zero prefetch-worker wins during prefill. `SLOW-PREAD` warnings count only reads above 200 ms: there were 161, and the immediate repeat still had 142. The earlier NaN run routed to only six experts per late layer, reading 8.61 GB total and appearing faster. Correct routing accounts for the increase in reads; SSD projection reads are now the throughput bottleneck. Full plan output: `build_mtp_results/2026-11-03-e216712-e216712-deepseek41-engram-same-layer-clean-smoke/model-capacity-plan.txt`.


## Why there were SSD reads
Full MoE residency without Engram is 157.13 GB across 15,360 experts. The smoke profile allows 85 GB host memory, caps DDR at 6,000 expert slots, and configures 21 GB MoE cache in VRAM; it does not set `ram_preload_all_experts`. Therefore the model is cache-on-demand: the 97-token prompt caused 4,271 experts / 36.87 GB to be read from SSD, with 0 DDR hits and 0 prefetch-worker wins. `SLOW-PREAD` is a warning for an individual projection read over 200 ms (161 occurrences), not the total read count. The prior faster run had NaN routing and only 1,080 expert blocks / 8.61 GB; its rate is invalid. A same-config warm repeat and async-H2D test both stayed around 0.74 tok/s prefill, with ~0.03 GB/s real SSD read bandwidth. Async upload moved 42.11 GB H2D but did not help because expert data fetches from SSD remain the bottleneck. Full capacity output: `build_mtp_results/2026-11-03-e216712-e216712-deepseek41-engram-same-layer-clean-smoke/model-capacity-plan.txt`.


## Host DDR limit for the requested 110 GiB cache
The current machine reports 125.48 GiB total RAM and 119.26 GiB MemAvailable. Retaining the 26 GiB emergency reserve leaves at most 93.26 GiB for new host allocations before ZeroMoE resident/scratch overhead. 110 GiB of MoE data plus the reserve alone requires 136 GiB of physical RAM; full MoE residency is 157.13 GB decimal. The current configuration was not raised and the guard was not weakened. To use a 110 GiB DDR expert pool safely, the host needs more physical RAM; on this machine, continue with the smaller on-demand DDR/VRAM cache and SSD-backed expert reads.

## Emergency reserve update
The DeepSeek starter and dual-GPU profile now stop new host allocations at 10 GiB MemAvailable (`mem_guard_stop_mb=10240`), as requested. Current host state was 125.48 GiB MemTotal / 119.26 GiB MemAvailable. A 110 GiB DDR payload plus a 10 GiB floor exceeds current MemAvailable by 0.74 GiB before runtime overhead, so no 110 GiB allocation was attempted. The live physical-memory guard remains enabled.

## Decode memory-plan root cause (2026-11-03)
`DeepseekV41GraphModel::plan_decode_memory()` currently returns `Unknown` after validating the architecture and topology. Its reason is that DeepSeek aliases engine-owned VRAM/DDR expert caches, but their placement is established by the later cache planner. `EngineRunner::run()` resolves graph policy before `plan_expert_cache()` and `setup_vram_expert_cache()` and does not resolve it again after those planning stages. Thus the runtime correctly keeps full-graph decode mandatory, but `check_full_graph_moe_run.py --full-fit` cannot pass. A fix must provide a pre-allocation, per-device byte estimate covering dense weights, planned MoE cache, KV/Engram state, and graph workspace; simply filling `required_device_bytes` from expert-cache capacity would be incomplete.


## DeepSeek V4.1 usage-ranked preload smoke (2026-11-03)

**Locally measured on the dirty `master` checkout** (`e2167121bef0748954a435e5a8d3fad8ccbb4969`), binary SHA `ebadd4a5a314aa330077e4083359e9ea2f483718f279854f67ef23c5b66a5c54`, GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`, CUDA 13.3, driver 610.57.04, native targets sm86/sm120. Persistent artifacts: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-usage-preload-test-retry4/`. This was a guarded smoke (97 prompt tokens, 8 generated, context 4096, temperature 0), not the final performance workload.

The full graph built and completed all 40 prefill layers and seven graph decode runs; the run gate reports zero one-token layered forwards and zero legacy per-expert dispatches. However, the memory plan remains `Unknown`, so `tools/check_full_graph_moe_run.py --full-fit` fails. `bench.json` reports prefill 5.73 tok/s, decode 1.06 tok/s, and decode wall 0.89 tok/s; the engine’s steady graph decode counter reports 0.78 tok/s. End-to-end active request rate was 0.30 tok/s. These are far below the user's 100 tok/s prefill / 10 tok/s decode targets. The 8-token run is too short to accept as a performance result.

The usage-ranked preload filled 6,000 DDR slots (51.59 GB) and 1,705 VRAM slots (20.98 GB), 7,705 slots total out of 15,360 (50.2% theoretical disjoint coverage). Prefill recorded 2,285 RAM hits, 572 main-thread claims, and zero prefetch-worker wins. Decode still spent 744.77 ms/token in SSD reads (`ram_batch` 746.72 ms/token; async upload 0.82 ms/token). This confirms that this 6,000-slot preload did not remove cold expert reads. The next speed work should focus on route-hotset coverage and making the graph’s DDR misses overlap with GPU work; the log warns the DeepSeek adapter lacks a common early-ready CPU lane.

The run completed its request with no NaN/Inf log report and no emergency stop (minimum MemAvailable 59.62 GiB), then exited `-6` during CUDA `cudaFree` teardown (`driver shutting down`). This smoke used an 85 GB host cap, 15 GiB pin budget, 20 GiB startup margin, and the requested 10 GiB runtime emergency floor. It does not verify the requested 110 GiB DDR configuration, which remains unsafe on this host. The GLM source-reported comparison and material configuration differences are in the artifact's `reference-settings.md`; this smoke is not a matched llama.cpp performance comparison.


## DeepSeek DDR-capacity A/B: 6,000 vs 8,000 slots (2026-11-03)

Locally measured, same 97-token prompt (same SHA), 8 generated tokens, context 4096, temperature 0, same binary (`ebadd4a5...`), same DDR usage files, host cap 85 GB, pin cap 15 GiB, emergency floor 10 GiB, and 11/10 GB VRAM caches. The only runtime setting changed was `ram_cache_total_experts` from 6,000 to 8,000. Artifacts: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-usage-preload-test-retry4/` and `...-retry5-8k-ddr/`.

| DDR slots | Prefill | Bench decode | Graph steady decode | Decode wall | SSD disk read/decode token | min MemAvailable |
|---:|---:|---:|---:|---:|---:|---:|
| 6,000 | 5.73 tok/s | 1.06 tok/s | 0.78 tok/s | 0.89 tok/s | 744.77 ms | 59.62 GiB |
| 8,000 | 6.36 tok/s | 1.07 tok/s | 0.74 tok/s | 0.84 tok/s | 727.67 ms | 43.73 GiB |

The 8,000-slot run preloaded 68.79 GB (64.06 GiB) DDR expert slabs plus 20.98 GB (19.54 GiB) in VRAM, for about 83.6 GiB of physical cache pools. It reduced main-thread prefill claims from 572 to 455, but prefetch-worker wins remained zero and decode still spent about 728 ms/token reading from SSD. Increasing DDR from 6,000 to 8,000 therefore did not solve decode. The unchanged 11/10 GB VRAM allocation is below the user's requested capacity; the next capacity profile will increase both DDR and VRAM while retaining startup admission and live 10 GiB emergency guards.

Both runs used full-graph prefill/decode (40 prefill layers, seven decode graph runs, zero layered one-token forwards or legacy expert dispatches), but both fail `check_full_graph_moe_run.py --full-fit` because the graph memory plan is `Unknown`. Both requests were numerically coherent with no NaN/Inf log, but teacher-forced parity remains open. Both processes exited -6 at CUDA cleanup (`driver shutting down`); neither hit the emergency memory stop. The 8-token measurements are smoke diagnostics, not the 10 tok/s decode / 100 tok/s prefill acceptance workload.


## DeepSeek 105 GiB DDR capacity attempt: VRAM headroom failure (2026-11-03)

Retry7 used the rebuilt binary with a DeepSeek-only 8 GiB startup margin, 8 GiB emergency floor, a 120 GB decimal process cap, 13,100 DDR expert-slot target, and 18.25/14.50 GB requested VRAM MoE caches. The byte-aware RAM planner capped the DDR pool at 12,560 slots / 107.998 GB (100.58 GiB) after resident and workspace/state reserves. VRAM pools allocated 32.1204 GB (29.91 GiB): 17.6249 GB on the 3090 and 14.4955 GB on the 5060 Ti. It reported maximum disjoint expert-identity coverage of 15,169/15,360.

Graph prefill failed before token generation: GPU0 had 0.97 GiB free, below the explicit 1.00 GiB graph allocation floor. MemAvailable minimum was 12.53 GiB, so the configured 8 GiB host stop did not fire. The process exits -6 during CUDA cleanup. This was not a performance result. The full-fit planner still reports `Unknown`; the run gate cannot accept `--full-fit`.

The host cannot provide a true 105 GiB DDR expert pool with the current resident/state/slack estimates: that requires about 124.65 GB process budget, plus the 8 GiB startup reserve, exceeding the observed available physical memory. Current effective safe DDR expert capacity is about 100.6 GiB. The next retry lowers the 3090's cache request by 0.5 GB to retain more graph headroom while preserving the large DDR pool and high 5060 cache.


## DeepSeek high-capacity preload retry8: swap-pressure abort (2026-11-03)

Artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-105gib-ddr-vram-headroom-retry8/`. Source `e2167121bef0748954a435e5a8d3fad8ccbb4969` (dirty), binary SHA `9e09ac1e284f287f759cb764224b66ab9d07133190c808886f01e54add703edb`, GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`. Both GPUs were idle at admission; initial MemAvailable was 120.42 GiB. Config requested 13,100 DDR slots (~105 GiB), startup/live reserve 8 GiB, and 17.75/14.4955 GB MoE VRAM pools.

The planner still capped DDR at 12,560 slots / 100.581 GiB (host budget 120 GB, resident 6.15967 GB, workspace/state slack 5.74826 GB). It allocated 17.6249 GB MoE on the 3090 and 14.4955 GB on the 5060 Ti; maximum disjoint coverage was 15,169/15,360. The model has `general.alignment=32`, so the run used buffered io_uring (`direct_io=false`, `drop_pages=false`).

The run was terminated after 302.9 seconds, before preload completion or token generation. The guard recorded minimum MemAvailable 19.01 GiB and did not trigger its 8 GiB stop, while swap usage climbed from ~2.1 to ~27 GiB. A host sample showed ~315 MiB/s swap-out, ~50% I/O wait, and memory PSI avg10 8.37%. SIGTERM did not promptly stop the model in `io_uring` wait; only the retry8 `zeromoe` PID was then SIGKILLed. After exit, swap usage returned to ~2.5 GiB and active swap-out stopped. No throughput result was produced.

This establishes that MemAvailable-only guarding is insufficient for this preload: the run entered severe reclaim/swapping well above the 8 GiB floor. Buffered source-page caching is a likely contributor, but not yet isolated. Next run requires `drop_pages=true`, continuous swap-out/PSI tripwires that stop before severe pressure, and an idle/recovery gate. Do not raise the DDR allocation cap until this is addressed. Independently, the engine emitted `[PLACEMENT-WARN]`: DeepSeek does not advertise a common early-ready CPU lane, so mixed VRAM/DDR execution serializes MoE branches and can explain a large decode penalty.


## Post-attempt correction and safety follow-up (2026-11-03)

Retry8's engine log shows the effective in-process MemAvailable floor was 10 GiB (`10240 MB`), because `graph_mem_guard_floor_mb` was still `-1`; setting `mem_guard_stop_mb=8192` alone did not lower the built-in 10 GiB minimum. The external Python watcher had an 8 GiB floor, but retry8's minimum was 19.01 GiB, so neither threshold fired. The active DeepSeek profiles now set `graph_mem_guard_floor_mb=8192` explicitly. They also set `drop_pages=true` so successful buffered io_uring reads advise source pages away.

The runtime guard now blocks further host allocations after two consecutive samples of high memory PSI or swap-out (full PSI >=2%, some PSI >=20%, or swap-out >=64 MiB/s). The new binary built from dirty source e2167121 has SHA `2cbed8d2b4fc8d0da21bde984ffbfd5d9347a031c392a5b1a1fbf71f7462881c`; the focused `host_budget_admission` component test passed. Retry8 remains an aborted preload with no token timing.


## DeepSeek retry9: drop-pages A/B and PSI threshold (2026-11-03)

Artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-105gib-ddr-vram-headroom-retry9/`. Retry9 used the rebuilt pressure-aware binary, `drop_pages=true`, explicit `graph_mem_guard_floor_mb=8192`, and the same 120 GB host cap / 13,100-slot DDR request / 17.75+14.4955 GB VRAM requests. Planner cap remained 12,560 slots (100.581 GiB); VRAM pools were 17.6249 and 14.4955 GB.

At 214.7 seconds the preload had filled roughly 57 GiB. MemAvailable was 58.94 GiB, `VmSwap=0`, swap-out was ~0.008 MiB/s, but memory PSI avg10 reached 2.34%. The first pressure guard blocked allocations at that point and the runner stopped the process. The host returned to ~120 GiB MemAvailable with both GPUs free. No generation or performance timing was produced. Compared with retry8 (27 GiB swap and ~315 MiB/s swap-out at the same stage), page dropping eliminated that swap behavior through the measured point. The 2% PSI-only cutoff was over-conservative under the SSD read workload.

Next change: retain an unconditional swap-out rate stop (64 MiB/s for two polls), but gate PSI thresholds on low MemAvailable (full PSI >=4% below 32 GiB; some PSI >=20% below 48 GiB). This keeps protection against the observed reclaim incident without stopping solely on read-related PSI while RAM is plentiful.


## DeepSeek retry10: early transient swap-pressure stop (2026-11-03)

Artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-105gib-ddr-vram-headroom-retry10/`. Retry10 passed idle admission after the `/tmp/opencode/survey2.toml` workload exited. `drop_pages=true`, actual 8 GiB engine floor, host cap120 GB, DDR request13,100 (planner12560 / 100.581 GiB), and VRAM caches17.6249/14.4955 GB were active. At 63.8 GiB process RSS, MemAvailable was53.746 GiB and this process `VmSwap` remained0. Global swap-out peaked at217.7 MiB/s and memory PSI at3.3%; runtime allocation admission blocked briefly and recovered. The external wrapper then stopped after two samples (exit -9), before the 8 GiB floor. No tokens or performance data were produced; host and GPUs recovered.

Read-only cgroup inspection showed `memory.max=max` for the user/session scopes; host `vm.swappiness=150`. This points to transient system reclaim rather than a 64 GiB cgroup cap. `drop_pages=true` improved on retry8's 27 GiB swap growth, but retry10's wrapper threshold is too aggressive. Next runner should let the allocator block allocations during short reclaim bursts and terminate only for sustained severe swap-out, process VmSwap growth, or a low MemAvailable/PSI combination.


## DeepSeek retry11: DDR slabs swapped after the 15 GiB pin budget (2026-11-03)

Artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-105gib-ddr-vram-headroom-retry11/`. The stable-idle gated run used the current binary, 8 GiB effective floor, `drop_pages=true`, 120 GB host cap, 13,100-slot DDR request (planner capped to 12,560 / 100.581 GiB), and 17.75/14.4955 GB requested VRAM caches. It reached 70.06 GiB RSS with MemAvailable 47.905 GiB, but process VmSwap reached2.002 GiB, cumulative global swap-out2.103 GiB, and PSI full/some avg10 peaked5.7%. It was stopped by the external cumulative-swap guard before the MemAvailable floor; no tokens or timing were produced. The host recovered.

Cgroup files show `memory.max=max`; host `/proc/sys/vm/swappiness` is150. With `ram_pinned_budget_gb=15`, most of the planned DDR pool remains pageable, consistent with ZeroMoE's own slabs moving into zram during preload. Retry12 will pin 105 GiB (enough for the planned 100.581 GiB pool) without increasing the host allocation budget; all 8 GiB and swap/PSI guards remain.


## DeepSeek retry12: 105 GiB RAM pin-budget A/B (2026-11-03)

Artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-105gib-ddr-vram-headroom-retry12/`. Retry12 kept the 8 GiB floor, `drop_pages=true`, 120 GB host cap, and planned 100.581 GiB DDR + 29.91 GiB VRAM pools, but raised `ram_pinned_budget_gb` from15 to105. At 66.84 GiB process RSS, MemAvailable was50.203 GiB, process VmSwap1.771 GiB, global swap-out2.075 GiB, and memory PSI4.15%. The runtime blocked allocations during swap bursts; the wrapper stopped at2 GiB cumulative swap, exit -9. No tokens/timing; host recovered. No pin failure warnings occurred, but shutdown counters did not run, so actual slab pin success is still unverified.

Pin budget alone did not avoid early swap; next attempt will log per-process VmPin/VmLck and allow a bounded larger swap total (8 GiB global / 6 GiB process) while preserving the 8 GiB MemAvailable floor and severe PSI/swap-rate stops.


## DeepSeek retry13: full DDR preload, graph VRAM headroom failure (2026-11-03)

Artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-105gib-ddr-vram-headroom-retry13/`. Retry13 used `drop_pages=true`, `ram_pinned_budget_gb=105`, the effective8GiB floor, host cap120GB, requested DDR slots13,100 (byte planner cap12,560 / 100.581GiB), and VRAM caches17.75/14.4955GB. Log confirms `[PIN] Preloaded 1024 VRAM-pinned + 640 VRAM-unpinned + 12560 RAM`.

MemAvailable minimum was10.636GiB, above the floor. Process RSS peaked100.971GiB; process VmSwap5.879GiB; global swap-out6.266GiB; memory PSI peaked8.05%. The VmSwap amount is near the planner's 6.15967GB resident-tensor accounting, so it may be host resident/dense copies rather than the MoE slab pool; no shutdown pin counters were captured, and the process VmPin metric did not show the CUDA registrations. The 105GiB pin budget was configured and no cudaHostRegister failure warnings appeared.

The request generated no tokens because the prefill graph refused to build at 0.80GiB free on the 3090, below its1.00GiB safety floor. Exit -6 during CUDA cleanup; no performance result. Retry14 lowers the 3090 cache to17.25GB, preserving the 5060 Ti and DDR settings, to restore graph headroom. The decode graph memory plan still reports `Unknown`; the graph path still warns that no early-ready CPU lane is advertised.


## DeepSeek retry14: PSI-only guard blocked before graph-headroom check (2026-11-03)

Artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-105gib-ddr-vram-headroom-retry14/`. The 3090 cache was reduced17.75->17.25 GB; the 5060 Ti remained14.4955 GB; DDR request13100, planner cap12560 /100.581GiB, pin budget105GiB, `drop_pages=true`, and actual8GiB engine floor. The wrapper stopped during DDR preload at81.59GiB RSS /30.79GiB MemAvailable when PSI reached8.17%. Process VmSwap peaked5.696GiB, cumulative swap5.860GiB; it did not hit the8GiB MemAvailable floor, and no graph or tokens ran. Host recovered.

Retry15 keeps the reduced 3090 cache, hard8GiB floor, and bounded swap limits, but does not stop on PSI alone while MemAvailable is high. That will allow us to validate the 1GiB graph floor after full DDR preload without relaxing the physical reserve.


## DeepSeek decode CPU/VRAM split lever (2026-11-03)

A code trace found an existing DeepSeek-specific decode split in `core/token_dispatch_pipeline.cpp`: when `cpu_offload_max_experts>0` on S=1, selected DDR experts outside the active-layer VRAM set can be forced onto the CPU executor while already-issued GPU expert kernels remain asynchronous. The DeepSeek route keeps per-expert outputs and accumulates in original k order, preserving the existing reduction tree. The benchmark profile had this knob at0, so the path was disabled. Retry15 now sets it to6 (the model's top-k) for an A/B. This is the shared cache-backed executor path; it does not make `DeepseekV41GraphModel::supports_cpu_lane()` true, and the full-graph memory plan remains `Unknown`. Measure with correctness checks before treating it as a performance improvement.


## DeepSeek graph CPU-cold path found (2026-11-03)

The existing `V41CacheMoeSite::cpu_cold_graph` path in `models/deepseek/ggml_deepseek_v41_graph.cpp` is enabled by typed `graph_cache_moe_cpu_cold`. It runs DDR-owned selected experts in an async CPU job and leaves VRAM-owned experts in the graph's GPU branches, then merges outputs. This is the full-graph CPU/GPU split; `cpu_offload_max_experts=6` is retained only for the cache-backed fallback. Retry15 config now enables the graph CPU-cold path. Its external prefill workload must exit before the retry runner can launch.
