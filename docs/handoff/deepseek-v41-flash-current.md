# DeepSeek V4.1 Flash handoff

Status as of 2026-11-07 (Australia/Sydney): **paused at the user's request**. No ZeroMoE model process is currently running. Resume from this worktree and do not infer that v53 passed because the model loaded successfully.

## Goal and acceptance

- Support DeepSeek V4.1 Flash with full-graph prefill and decode, CPU offload for DDR-resident MoE, all model experts resident in DDR or VRAM, and no duplicate graph-owned expert allocation.
- User targets: around **100 tok/s prefill** and **10 tok/s decode**. The project's full-graph onboarding rule also requires at least 90% of a matched llama.cpp decode result, a successful full-fit graph run, and teacher-forced correctness checks against llama.cpp.
- Do not test H2D decode misses. The latest profile keeps decode misses on CPU. Prefill may stage experts from DDR into already-allocated VRAM slots.

## Worktree and binary

- Worktree: `/home/ray/Projects/ZeroMoE/.worktrees/deepseek-v41-continuation`
- Branch/HEAD: `codex/deepseek-v41-continuation` / `47ef7b30ed0e7d3230cb56e252072f8a8387e573`
- The worktree is dirty and contains uncommitted DeepSeek changes plus earlier unrelated Mimo artifact deletions. Preserve those changes; do not reset or switch this work to master.
- Last checked binary: `build_dsv41_retry16/zeromoe`, SHA-256 `2846094c7dfeb3ff0b7476371b85611a8c3142d2a550355ecc572e6b2735bf08`. It passed `tools/build_provenance.py check` before this handoff document was added; rerun the check and rebuild/restamp if needed before any performance run.
- Build provenance: GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`; applied GGML patch manifest SHA-256 `5c4c17e7164237cf7da1570a5dcf1194fbd700df0ee3ca980bf96dcc4530f95b`; CUDA 13.3.73; NVIDIA driver 610.57.04; GCC 16.2.1; native targets 86 and 120.

## Model and v53 test profile

Model source: `/mnt/usb-ssd/DeepSeek-V4.1-Flash-1017-GSQ-RCO/`. It contains five main IQ2_XXS shards plus `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` as component six. The current mount was verified as `/dev/sdd1` (exFAT, read/write), and a 4 KiB GGUF read succeeded. v53 uses symlinks only; no model payload was copied into the worktree.

The guarded v53 profile is in `build_mtp_results/2026-11-05-47ef7b3-deepseek41-cpu-prefill-handoff-alignfix-v53/`:

- Full graph prefill/decode, `ram_preload_all_experts=true`, strict full MoE residency.
- `host_ram_gb=119.5`, 4 GiB startup margin, 5 GB request headroom, 4.5 GiB engine floor, and `MemorySwapMax=0` for the model scope.
- 128 preallocated shared dynamic VRAM slots. Permanent VRAM pins use the frozen decode usage file; `prefill_pin_file` is empty so prefill survey volume cannot displace decode-hot pins. All unpinned model experts remain resident in DDR.
- CPU decode misses (`deepseek_v41_decode_moe_strategy="cpu"`). No H2D decode request was sent.
- Workload: the same 959-token prompt and 64 generated tokens. The runner stops after its first failed request and does not retry.

## Latest v53 run: loaded, then failed before prefill

The v53 admission gate passed three samples: MemAvailable 130.49–130.53 GB against 128.795 GB required, both GPUs visible and idle, SwapUsed 1.281 GiB, no measured swap-out. The model then loaded fully:

```text
[FULL-RESIDENCY-CHECK] resident=15360/15360 identities; DDR=11749, VRAM=3611, missing=0
```

The alignment preflight also passed on both GPUs (`base_mod16=0`). Startup's decode-only pin profile logged 95.14% average historical route coverage; this is a startup estimate, not request hit rate.

The first CPU-only request returned HTTP 500 after 13.149 seconds. It processed zero prompt tokens and generated zero tokens. The error was:

```text
layer=0 GPU=0 dummy preflight failed: no unpinned dynamic slot on this GPU;
pool_slots=128 ready_same_layer=0 pinned_DDR_candidates=228 local_dynamic_slots=0 compact=1
```

Therefore v53 has **no measured prefill or decode speed** and did not pass the full-graph run checker. The engine logged that it was keeping the server alive, but a later health request failed and the process was absent at the current check; no model is now loaded. Run evidence is under the v53 artifact directory, especially `server.log`, `request-cpu-result.json`, `response-cpu.json`, `host-process-memory.csv`, `system-io.csv`, `wait-gate-result.txt`, and `full-residency-plan` log entries.

The guarded load remained within the configured safety limits: MemAvailable low-water 19.461 GiB, process VmSwap 0, SwapUsed 1.281 GiB during the request window, and no emergency stop. The process read about 129.1 GiB while loading from the USB drive; the request failed before any decode, so that load I/O is not evidence of cold SSD reads during decode.

## Previous successful speed baseline

The last successful full-residency CPU run is v51 at `build_mtp_results/2026-11-05-47ef7b3-deepseek41-cpu-fullresident-grouped-v51/`. Two 959+64 requests measured:

| Request | Prefill | Decode | VRAM route hits |
|---|---:|---:|---:|
| CPU 1 | 12.16 tok/s | 2.22 tok/s | 68.1% |
| CPU 2 | 11.54 tok/s | 2.28 tok/s | 66.0% |

CPU expert workers accumulated 13.65–14.14 seconds per request, with 12.99–13.44 seconds of merge wait. Sampled decode GPU utilization averaged 6.9–8.4% on the 3090 and 1.3–2.2% on the 5060 Ti. This points to CPU DDR expert execution and merge barriers as the measured decode bottleneck; it is far below the 10 tok/s target. v51 passed the graph-run checker, but did not pass performance acceptance or correctness/reproducibility acceptance.

## Fixes present in the current worktree, not yet validated end to end

1. Shared compact VRAM pool placement aligns the dynamic tail down to 16 bytes. The prior decimal-sized GPU0 allocation left the tail 8 bytes off alignment. A startup check now validates pool base, slot offsets/strides, alignment, and bounds before DDR preload.
2. Prefill CPU spill merging now iterates over the graph's `D x S` element count. The old code used `site.cpu_output.size()`, which is empty on the prefill scratch-buffer path and omitted CPU spill contributions.
3. The per-layer CPU/GPU merge node releases that layer's VRAM compute holds after its merged output is ready, so later graph layers can reuse the shared dynamic LRU slots.
4. The v53 profile excludes prefill survey counts from permanent VRAM pin ranking. Offline route replay found that the frozen survey had 690,480 prefill selections versus 45,360 decode selections (15.22x). With no compact per-layer byte caps applied, top-3,611 decode-survey experts covered 80.668% of captured decode routes; a prefill-merged estimate covered 62.483%, near v51's measured 66–68% hits. Ranking the exact two-request route set covered 96.634%, likely overfit. See `route-replay-estimate.txt`.
5. Focused compact-cache/alignment/hold-release and grouped CPU IQ1_M tests pass. These unit tests do not exercise the loaded full graph.

## Remaining work and resume order

1. Diagnose why layer-0 prefill sees zero unpinned shared slots despite a 128-slot reserve and a successful request-boundary reset. Add logs/tests for each slot's `pool_layer`, current identity, hold count, pin state, route owner/pool match, and unpinned-index membership immediately before and after prefill staging. Ensure a valid same-layer dummy is available without consuming all staged slots.
2. Rebuild and pass the focused tests/provenance check. The handoff file changes the worktree snapshot, so refresh the binary manifest before performance runs.
3. With the USB source mounted and the host/GPU admission gates passing, run one guarded v53 request. If it succeeds, inspect per-layer `[DSV41-CACHE-MOE-RELEASE]` counts, full residency, CPU spill merge, route hit/miss, prefill staging, and any cold SSD reads before allowing the second request.
4. Then run matched long prefill/decode measurements, `check_full_graph_moe_run.py --full-fit --require-full-graph-prefill`, teacher-forced route/hidden-state comparisons against llama.cpp, and `check_full_graph_moe_perf.py` with the exact workload and `--minimum-tps 10`. Prefill 100 tok/s, decode 10 tok/s, and matched llama.cpp performance acceptance all remain open.

Current machine snapshot at handoff: no ZeroMoE process; MemAvailable about 119 GiB, SwapUsed about 1.0 GiB; the last `nvidia-smi` check was intermittent. The model files are currently readable at the path above, but a fresh guarded admission check is required before loading.

## Update 2026-11-07 (v56): shared-slot failure fixed, first measured rates

- Root cause of the v53 layer-0 dummy failure: startup pinning (`upload_expert(..., startup_plain_copy=true)`) took the 128 shared dynamic slots (`pool_layer == -1`) and pinned them, leaving 128 unpinned per-layer slots. The reset check only counted unpinned slots, so it passed. Fix: `evict_lru(..., skip_shared_dynamic)` excludes shared slots for startup pins, and `reset_unpinned_expert_cache` now requires every unpinned slot to be in the shared reserve. Failure path also logs a `slot census` line.
- `host_ram_gb` must be at least about 118 for full residency plus 5 GB headroom (115 fails: safe budget 103.09 GB < 106.03 GB). v56 uses 118.0 (gate needs about 127.3 GB MemAvailable on a 131.6 GB host).
- v56 (`build_mtp_results/2026-11-07-47ef7b3-deepseek41-cpu-prefill-pinfix-v56/`, binary sha256 d0d0e9c6...09e6, dirty tree on 47ef7b3): 959-token prompt, 64 tokens, CPU decode misses, one request: **prefill 31.15 tok/s (30.8 s), decode 3.63 tok/s**. Single request, no llama.cpp comparison, no teacher forcing. `check_full_graph_moe_run.py` FAILS: no `[MOE-GRAPH-GATE]` summary in the log.
- Goal (user, 2026-11-07): decode about 10 tok/s, prefill above 70 tok/s.

### v56 warm-request analysis and v57 status (2026-11-07)

- v56 STATS: seq 1 (cold) prefill 31.15 tok/s, decode 3.63; seq 2 (warm) **prefill 66.48 tok/s, decode 3.77 tok/s** (p50 262 ms/token). Cold prefill lost about 7 s at layer 7 to first-touch pages.
- Decode per-layer timing (warm): layers with zero misses still take about 4.5-5 ms, a floor of about 190 ms/token (about 5.3 tok/s) before CPU waits. Layers with misses add the CPU wait (merge_wait about equal to worker_ms; GPU idle while the CPU runs).
- The decode graph is 5438 nodes / 281 splits (about 7 splits per layer): the per-layer route node and CPU merge node force backend switches and host syncs. No CUDA-graph use on this path. Reaching 10 tok/s needs about 2.5 ms/layer, so the split/sync structure is the main lever, ahead of CPU kernel speed.
- Single-expert CPU misses used the scalar path (4-9 ms) because `quant_grouped` required count >= 2. v57 binary (sha256 29ec1c48...68ca) relaxes that in `core/cpu_expert_executor.cpp` and `ggml_deepseek_v41_graph.cpp`. **Not measured**: the USB SSD re-enumerated as /dev/sde1 mid-run (stale /mnt/usb-ssd on sdd1, I/O errors) and root is needed to remount. This changes CPU rounding for those experts; it needs the teacher-forced check before it is kept.
- Next: remount, rerun v57 (two requests), compare with v56 warm; then profile/merge the per-layer splits.

### Where the decode per-layer floor comes from (code reading, 2026-11-07; unmeasured)

- Not the HC nodes: `decode_gpu_hc` is on by default and v56 logged "[GGML-DSV41-DECODE] hyper-connections on GPU".
- Remaining CPU-backend custom nodes per layer in `models/deepseek/ggml_deepseek_v41_graph_decode.cpp`: attention (`swa_node`, `qkv`, `context`, `out`, lines about 556-603) and `moe_node`/`engram_node` (about 676-701). Each pinned node is a backend split with a host sync; this matches 281 splits / 5438 nodes for 40 layers.
- Next step once the drive is back: confirm with a per-split timing (ggml sched profiling or nsys) before changing anything; the likely work is moving the qkv/context/out stages to GPU ops. Any such change needs the teacher-forced same-input A/B from CLAUDE.md section 2 because attention rounding can flip routed experts.

### v58 measurement of the v57 binary (grouped single-expert CPU misses), 2026-11-07

Run: `build_mtp_results/2026-11-07-47ef7b3-deepseek41-cpu-grouped1-v58/` (USB SSD remounted as /dev/sde1; same 959+64 workload, host_ram_gb=118.0). Three requests in the server log:

| Request | Prefill tok/s | Decode tok/s | Decode p50 ms |
|---|---:|---:|---:|
| 1 | 65.90 | 4.00 | 244 |
| 2 | 66.21 | 4.21 | 231 |
| 3 | 66.47 | 4.20 | 234 |

About +11% decode versus v56 warm (3.77). Prefill unchanged (about 66). Goal (decode about 10, prefill above 70) still unmet. The runner's own second request was skipped by the swap-out guard; no teacher-forced comparison was run, so the grouped single-expert rounding change is not yet accepted. Server stopped afterwards.

### Decode 4.2 -> 7.7 tok/s (2026-11-07; v59-v63, all single-profile, dirty tree, UNVERIFIED numerics)

Profile (v59, new attention split log): host attention was 147 of 240 ms/token; of that the grouped out-projection node was 109 ms.

| Run | Change | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| v58 | grouped single-expert CPU misses | 66 | 4.0-4.2 |
| v60 | Q5_K `attn_output_a` out-projection on GPU | 39-47 (Engram stall) | 6.6-6.8 |
| v61 | parallel Engram miss fetch + `graph_engram_cache_mb=128` | 65.4-66.5 | 7.3 |
| v62 | `OMP_WAIT_POLICY=active` (REJECTED: worker 3.6 -> 5.0 ms, decode 6.66) | 66 | 6.7 |
| v63 | IQ2_XS grouped CPU dot (7 layers no longer scalar) | 65.7-66.4 | **7.7** (p50 126 ms) |

- Root causes: (1) `attn_output_a` is Q5_K but the GPU grouped path only accepted Q2_K, so every layer ran a 134 MB F32 CPU dot; fix = `CudaManager::grouped_output_proj_dense_host` (per-group F32-activation matvec). (2) Engram rows were read one at a time from the USB SSD with `posix_fadvise(DONTNEED)` and an 8192-row cache: about 11 s per request on a cold read; fix = parallel pread of distinct missing rows. (3) Layers 17/19/27/34/35/38/39 use IQ2_XS gate/up which had no grouped kernel.
- Targets still unmet: decode about 10, prefill above 70. Remaining decode: CPU expert misses (about 2.2 ms/expert, GPU waits), compressed context node 42 ms/token, qkv source node 17 ms/token.
- Numerics NOT validated: Q5_K GPU out-projection, grouped single-expert, and IQ2_XS grouped paths all change rounding. Compare routed experts/hidden states teacher-forced vs llama.cpp (CLAUDE.md section 2) before accepting.

### Route bisect and thread-count test (2026-11-07)

- Decode route-ID comparison, same prompt, run-to-run control v58 vs v59 = 0/2520 differing lines (deterministic). Prefill routes identical across all runs. Each rounding change flips decode routes: grouped single-expert CPU kernel 175/2520, Q5_K GPU out-projection 149/2520, IQ2_XS grouped dot 62/2520. This does not say which side is closer to llama.cpp; the teacher-forced llama.cpp comparison (reference at ~/Projects/llama.cpp, pinned 8baabd53, `scripts/validate_vs_llama.sh`, needs an idle host and loads the whole model on CPU) is still required before accepting these paths. Decision pending with the user: validate now vs optimize first.
- v63 miss profile: only 4.8% of decode routes miss and 22% of layer-steps have any miss (about 30 of 126 ms/token), so better pinning has limited headroom. Host attention (context node about 42 ms/token, per-head loop about 0.9 ms/layer) is the larger remaining cost and needs a GPU port with its own parity check.
- CPU is a 6-core/12-thread Xeon W-2235. `OMP_NUM_THREADS=6` (v64): decode 7.98 tok/s (+3.5%) but prefill 62-63 tok/s (-5%); not adopted. `OMP_WAIT_POLICY=active` (v62) was worse on both. A per-phase thread count (prefill 10, decode 6) might get both; untested.

### Per-phase CPU worker threads (v65, 2026-11-07)

The CPU-expert worker is a fresh `std::async` thread, so `omp_set_num_threads` there only changes its own team. Decode misses now use `hardware_concurrency()/2` (6 here); prefill spill keeps the process default (10). v65 (OMP_NUM_THREADS env 10): prefill 65.75 / 66.56 tok/s, decode 7.95 / 7.91 tok/s (p50 123 ms) versus v63 66.4 / 7.7. The env override used for the A/B was removed afterwards (the rebuilt binary has the fixed policy; not re-measured). Note: `main` calls `omp_set_num_threads(hardware_concurrency-2)` which overrides `OMP_NUM_THREADS` for the main thread, so the v64 env test only affected worker threads.
- Prefill idea not yet tried: per-layer H2D staging is 128 experts x 12.3 MB = 1.57 GB at about 142 ms/layer (about 5.7 s of the 14.4 s) and is not overlapped with the previous layer's compute; double-buffering the shared dynamic slots (+1.5 GB VRAM) and staging layer L+1 during layer L is the lever for the last 0.8 s.

### Prefill goal reached with 160 shared dynamic slots (v66/v67, 2026-11-07)

PCIe is the prefill limiter: the 3090 is Gen3 x16 and per-layer staging (128 experts, 1.57 GB, about 142 ms) already runs at link speed, and the 5060 Ti is Gen3 x8. Changing how many experts are staged per layer trades PCIe time against the CPU spill lane:
- v66, `full_moe_vram_dynamic_slots=96`: prefill 52.3 / 52.9 tok/s (worse; CPU spill cannot hide), decode 8.03 / 8.05.
- v67, `full_moe_vram_dynamic_slots=160`, `host_ram_gb=118.6`: **prefill 70.85 / 72.12 tok/s**, decode 7.92 / 7.95 tok/s. 160 slots leave 3566 VRAM pins (was 3611), so the DDR complement grows to 101.42 GB and 118.0 fails the safe-budget check by 0.33 GB; 118.6 passes. The guarded gate then needs about 127.9 GB MemAvailable on a 131.6 GB host (about 0.3-0.6 GB margin).
- Run: `build_mtp_results/2026-11-07-47ef7b3-deepseek41-slots160-v67/`. Same 959+64 workload, two requests, binary 236c310b... (dirty tree on 47ef7b3). Numerics paths (grouped single-expert, Q5_K GPU out-projection, IQ2_XS grouped) still not validated against llama.cpp.
- Decode still about 7.9 tok/s (goal about 10): remaining cost is host attention (context node about 42 ms/token).

### attn_score_threads sweep (v68/v69, 2026-11-07; profile already had 8)

Same v67 profile (160 dynamic slots, host_ram_gb 118.6). attn_score_threads 6: decode 7.38, context node 48 ms/token. 8 (baseline v67): decode 7.92-7.95, context node 42. 12: decode 7.23-7.29 although the context node drops to 39 ms (extra threads contend with the rest of the graph). Prefill unchanged (about 71-72). Keep 8. Thread-count tuning is exhausted.
- The per-head loop is compute-bound: about 640 keys (window + top-512 compressed) x 64 heads x 512 dims per layer, about 0.4 ms at 6-core FMA peak vs 0.9 ms measured. A GPU port needs a device mirror of the compressed KV (`state.keys`) and window cache (`KVc`) to avoid paying the savings back in H2D copies (about 1.3 MB per layer per token); estimated gain 0.9 -> about 0.25 ms/layer, i.e. about 26 ms/token (122 -> about 96 ms = about 10.4 tok/s). Needs a route-ID comparison against v65 at each step.

### Performance goals reached (v70-v74, 2026-11-07) -- NOT accepted

Goal: decode about 10 tok/s, prefill above 70 tok/s. Final binary `build_dsv41_retry16/zeromoe` sha256 b3149692a63f44620b13494544ee45372ed8aee22d069df48f44da99d83d2c90 (dirty tree on 47ef7b3), run `build_mtp_results/2026-11-07-47ef7b3-deepseek41-final-confirm-v74/`, profile = v67 (160 shared dynamic slots, host_ram_gb 118.6, attn_score_threads 8, graph_engram_cache_mb 128), 959-token prompt + 64 generated tokens, three requests:

| Request | Prefill tok/s | Decode tok/s | Decode p50 |
|---|---:|---:|---:|
| 1 | 70.37 | 10.73 | 90.6 ms |
| 2 | 71.81 | 10.81 | 91.5 ms |
| 3 | 71.66 | 10.81 | 91.0 ms |

Changes since v63 (decode 7.7): per-phase CPU worker threads; per-head compressed-layer attention on the GPU (`kernels/models/dsv41_head_attention.cu`, `CudaManager::dsv41_head_attention_host`) with per-layer device mirrors of the window and compressed keys (incremental upload, invalidated at request start in `dsv41_attention_impl`); Engram miss fetch parallelised for every lookup of 8+ rows including single-token decode.
- Honest-number caveat: requests 2 and 3 replay the same tokens, so the Engram row cache hides disk reads; request 1 (novel tokens, cold rows) is the representative number. Before the decode Engram fix, request 1 was 9.5 tok/s (`hc_engram` 12.1 ms/token) while requests 2-3 showed 10.8; after it all three agree.
- Route check: v72 request 1 decode routes == v70 (0/2520), so the device mirror is exact. Request 2 differs from request 1 by 110-160 route lines in every build including CPU-only ones, so only request 1 is a clean comparison.

NOT done (acceptance gates in CLAUDE.md section 2 still open):
1. Teacher-forced comparison against llama.cpp (reference at ~/Projects/llama.cpp, pinned 8baabd53, `scripts/validate_vs_llama.sh`). Four rounding-changing paths are unvalidated and each flips 2.5-7% of decode route lines: grouped single-expert CPU kernel, Q5_K GPU out-projection, IQ2_XS grouped dot, GPU head attention.
2. `check_full_graph_moe_run.py --full-fit --require-full-graph-prefill` now PASSES on the v74 log (it failed on the v56 log). It checks graph policy/counters only, not correctness or speed.
3. Matched llama.cpp performance (>= 90% bar) and `check_full_graph_moe_perf.py --minimum-tps` not run.
4. Nothing is committed. Run-to-run noise: single profile, one machine, three requests.

## Decode profile of the 91 ms token (v74 log, 2026-11-07)

Source: `build_mtp_results/2026-11-07-47ef7b3-deepseek41-final-confirm-v74/server.log` (request 3; 63 decode runs, warm mean 92.4 ms/token). Timers are host-side per node; MoE numbers come from the per-layer `[DSV41-CACHE-MOE]` / `[DSV41-CACHE-MOE-CPU]` log lines.

| Bucket | ms/token | Note |
|---|---:|---|
| attention_host total | 31.6 | context node 16.6 (GPU head attention incl. staging copies), out-projection 8.2, SWA/qkv-host 6.8 |
| native_qkv (Q/KV dense projections) | 15.6 | about 0.39 ms/layer, each a host-to-device round trip |
| MoE CPU merge wait | 13.0 | 11.65 cold experts/token in 9.0 layers; CPU worker 16.8 ms/token (1.44 ms per cold expert), only ~4 ms overlapped |
| MoE route + upload | 4.8 | 3.35 route, 1.43 upload |
| HC/Engram + logits | 1.8 | |
| Unattributed (GPU graph compute for resident experts, shared expert, norms, sync) | ~25 | derived: 92.4 minus the rows above; not directly measured |

Findings:
1. About 47 ms (attention 31.6 + qkv 15.6) is 120 small host-to-device round trips per token (3 per layer), each ~0.4 ms: latency-bound, not compute-bound. Making Q/KV projection, attention context and out-projection one GPU-resident chain per layer (no host staging), ideally CUDA-graph captured, is the largest lever (estimated 20-30 ms).
2. The CPU cold-expert lane costs 13 ms of exposed wait. Launching it earlier (route prediction / pilot) or pinning more of the hot set in VRAM would hide most of it (estimated 8-10 ms).
3. The ~25 ms unattributed bucket needs a per-node GPU timer (CUDA events) before tuning.
Estimated reachable decode with 1+2: about 55-60 ms/token (16-18 tok/s) if the estimates hold; not measured. Any new GPU attention/projection path changes rounding and needs the llama.cpp teacher-forced check.

## v75: fused Q/KV projection chain (commit 0709643), measured 2026-11-07

Artifact: `build_mtp_results/2026-11-07-0709643-deepseek41-fused-qkv-v75/` (binary built from clean 0709643, sha256 614287c0...ce60b9... see `zeromoe.build-info.json`; same config as v74 except `host_ram_startup_margin_gib = 3.0` because a shared host left only 127.1 GB available against the 127.9 GB gate; request headroom 5 GB unchanged). Two requests (the harness sends two), same 959+64 workload.

| | v74 | v75 |
|---|---:|---:|
| Prefill tok/s | 70.37 / 71.81 / 71.66 | 69.91 / 71.86 |
| Decode tok/s (first-to-last) | 10.73 / 10.81 / 10.81 | 10.95 / 11.07 |
| Decode p50 ms | ~91 | 89.8 / 87.7 |
| native_qkv ms/token | 15.56 | 12.56 |
| warm decode mean ms | 92.4 | 91.6 |

- Decode routes are bit-identical to v74: 0 of 2,520 route lines differ in each of the two requests (as designed: same kernels, same order, no host hops between them).
- `check_full_graph_moe_run.py --full-fit --require-full-graph-prefill`: PASS.
- Gain is small (about +2% decode, native_qkv -3 ms). Request-1 prefill (69.91) is just under the 70 tok/s target; two requests on a shared host are not enough to call it a regression or a gain. The attention timers rose (context 16.6 -> 18.0, other 6.8 -> 9.0 ms), so some of the saving is eaten elsewhere or is run-to-run noise; v74 baseline is left unchanged.
- Fusing only saved 3 of the estimated 5-8 ms; the remaining Q/KV cost is kernel time, not hop latency. Next: nsys to split the ~25 ms unattributed bucket before fusing attention.

## v76-v78: nsys profile and Q5_K attn_q_b kernel (2026-11-07)

**nsys (v76, `build_mtp_results/2026-11-07-0c585d1-deepseek41-nsys-v76/decode.nsys-rep`, capture covers prefill and decode of two requests).** The plain per-element `q5_K_kquant_matmul_kernel` ran 5,040 times on the RTX 3090 (grid O=32,768, S=1; one call per layer per token = `attn_q_b`, I=1024 so 4 blocks/row, below the `lane8` threshold of 8) at 250.6 us/call: about 10 ms/token, roughly 10% of memory bandwidth. Other large items in the capture (mostly prefill): ggml `mul_mat_q` for IQ types, `dequantize_block_iq1_m` (16,419 calls, 733 ms), `q5_K_kquant_matmul_kernel_lane8` (small shapes), `dsv41_head_attn_kernel` 130 us/call (4,788 calls, about 5 ms/token). `cudaStreamSynchronize` is 52.7% of CUDA API time (220,771 calls).

**v77 (commit 8a132a0, reverted as ec264ca).** Unrolling the 4-block loop in the per-element kernel (bit-identical arithmetic) changed nothing: routes 0/2,520 different, `native_qkv` 12.48 ms. The kernel is limited by per-element dequant work, not load latency.

**v78 (commit ed1c9a5): Q5_K uses the `lane8` kernel for 4-7 blocks/row.** Two requests, same 959+64 workload and config as v75 (startup margin 3.0 GiB):

| | v75 | v78 |
|---|---:|---:|
| Decode tok/s | 10.95 / 11.07 | **11.63 / 11.91** |
| Decode p50 ms | 89.8 / 87.7 | 84.2 / 82.0 |
| native_qkv ms/token | 12.56 | **6.82** |
| Warm decode mean ms | 91.6 | 83.9 |
| Prefill tok/s | 69.91 / 71.86 | 70.96 / 71.92 |

- `check_full_graph_moe_run.py --full-fit --require-full-graph-prefill`: PASS. Generated text starts identically and is coherent.
- This is a rounding change (different reduction tree): 438 and 407 of 2,520 decode route lines differ from v75 in the two requests (free-running, so later divergence compounds). Not compared with llama.cpp. Treat as a candidate: v74 stays the documented baseline until the teacher-forced check is run.

## v79-v82: model files on the local NVMe, head-attention interleave (2026-11-07)

- **Model location.** The six GGUFs were copied (user-approved) from the USB SSD (`/mnt/usb-ssd/DeepSeek-V4.1-Flash-1017-GSQ-RCO/`, exFAT) to the local NVMe `/home/ray/vllm-workspace/DeepSeek-V41-Flash/GSQ-RCO-IQ2_XXS/` (sizes verified equal; Engram first 2 GB compared). **Server start to `/health` dropped from about 324 s to about 62 s.** `guarded_start.sh` `MODEL_DIR` points at the NVMe folder in v79+.
- **v79** was killed by the host-pressure watchdog 54 s after launch (swap-out 165 MiB/s, MemAvailable 39 GB during preload, `emergency-stop.json`); the same config loaded fine in v80-v82. Cause not isolated (page-cache pressure from the faster disk or a neighbor's load at that moment).
- **Head-attention interleave (two output chains, 4-key unroll; commit cad5cf0, reverted as 77ba56f).** Bit-identical to v78 (0 of 2,520 route lines differ in both requests) but no speedup: `context_node` 17.5 ms vs 17.3, so the kernel is not the bottleneck of that node. Removed.
- **NVMe runs (same binary as v78 except the reverted kernel; 959+64 workload, 2 requests, startup margin 3.0):**

| Run | Disk / IO | Prefill tok/s | Decode tok/s | Decode p50 ms |
|---|---|---:|---:|---:|
| v78 | USB, buffered | 70.96 / 71.92 | 11.63 / 11.91 | 84.2 / 82.0 |
| v81 | NVMe, direct_io=true | 68.82 / 70.64 | 11.33 / 11.77 | 86.0 / 83.0 |
| v82 | NVMe, buffered | 68.32 / 70.74 | 11.07 / 11.72 | 88.2 / 82.6 |

Routes in v81 are identical to v78 (0/0), so the disk does not change numerics. Throughput is about 2-3% lower than v78 in both NVMe configs; attention host-side "other" rose 6.3 to 8.4-8.6 ms. The cause is not isolated (host state differed between runs; only two requests each), so v78 stays the best measured number and no baseline is changed. `check_full_graph_moe_run.py` PASS in all.

## Pending measurement: batched out-projection and GPU indexer scores (commits 4aaee8a, 446f685)

Both are built into `build_dsv41_retry16/zeromoe` (source 446f685) and are designed to be bit-identical to v78:
- `4aaee8a`: one launch for all Q5_K grouped out-projection A groups (`launch_q5_K_matmul_grouped`). The v84 run (`build_mtp_results/2026-11-07-grouped-outproj-v84/`) was contaminated by another session's zeromoe (host swap-out; request 1 decode 6.77 tok/s, request 2 skipped), so it gives no timing. Its request-1 decode routes matched v78 (0 of 2,520 lines differ).
- `446f685`: Lightning-indexer block scores on the GPU for single-token decode (`kernels/models/dsv41_indexer_score.cu`, mirror of the write-once indexer keys per layer). The engine builds with `-std=c++23`, which disables FMA contraction, so the kernel uses separate multiply and add; `build_dsv41_retry16/dsv41_indexer_score_test` compares it bit for bit with the host loop (1, 7, 64, 260, 1000 blocks: PASS). Host cost being removed: about 0.7 ms per CSA-layer call, roughly 5.7 ms/token.
- Estimated combined gain 4-7 ms/token (unmeasured). Gate: v85 two-request run on a quiet host, 0 differing decode route lines versus v78 (`2026-11-07-lane8q5-v78/observed-routes.log`), `check_full_graph_moe_run.py` PASS.

nsys facts used (v83, `--cuda-graph-trace=node`, ~101 ms/token under profiling): RTX 3090 busy 27%, RTX 5060 Ti 6%; about 800 host-to-device and 525 device-to-host copies per token; lane8 Q5_K matvecs 10.9 ms/token on the 3090 (mostly the 8-group out-projection), head attention 5.3, expert `mul_mat_vec_q` 5.4 (3090) + 5.4 (5060 Ti).

## v85: batched out-projection + GPU indexer scores measured (2026-11-07)

Run `build_mtp_results/2026-11-07-outproj-indexer-v85b/` (binary from source fd320bd = 446f685 code; model files on NVMe, direct_io=true, startup margin 3.0 GiB; host quiet during the run; two requests, 959+64):

| | v78 (USB) | v81 (NVMe, same kernels minus the two new changes) | v85b |
|---|---:|---:|---:|
| Decode tok/s | 11.63 / 11.91 | 11.33 / 11.77 | **12.10 / 12.66** |
| Decode p50 ms | 84.2 / 82.0 | 86.0 / 83.0 | 80.6 / 76.4 |
| Warm decode mean ms | 83.9 | 88.6 | 83.1 |
| Prefill tok/s | 70.96 / 71.92 | 68.82 / 70.64 | 68.13 / 69.93 |
| attention context_node ms/token | 17.3 | 18.7 | 15.2 |
| attention outproj_node ms/token | 8.2 | 8.5 | 7.5 |

- Gain versus the same NVMe setup (v81): decode +7% (11.33/11.77 to 12.10/12.66); smaller than the 4-7 ms/token estimate, `context_node` -3.5 ms and `outproj_node` -1.0 ms. Prefill is unchanged within noise but about 2-3% below the USB-disk v78 in both NVMe runs (cause not isolated).
- Routes: request 1 is bit-identical to v78 (0 of 2,520 decode route lines differ). Request 2 has 256 differing lines, first at token 6, layer 26 (one near-tie expert), then cascading. Not yet explained: either run-to-run non-determinism in request 2 (expert placement after prefill) or a state bug in one of the two new paths on a second request. The determinism rerun (v85c) was ruined by a neighbor's compile (HTTP 408 timeout; its partial request-1 lines still matched v78). Rerun the same binary and compare request 2 between two runs before calling the new paths bit-identical across requests.
- `check_full_graph_moe_run.py --full-fit --require-full-graph-prefill`: PASS.

Determinism reruns of v85b (v85c, v85d, v85e) are invalid: v85c hit HTTP 408 under a neighbor's compile; v85d was a duplicate-client/stale-runner mishap; v85e was stopped by the request swap-out guard (neighbor compile plus model preload). The request-2 route question (256 lines from token 6, layer 26) stays open and needs a run on a host with no other builds or runs for about 10 minutes.

Offline analysis of the request-2 question from existing logs (no new run): request 2 already differs from request 1 inside every build, at the same first line: v78 320 of 2,520 decode route lines (first at token 5, layer 30), v81 320, v85b 366 (first line identical, 230). At the v85b flip (token 6, layer 26) v78 request 1 has expert 93, v78 request 2 has 121, and v85b request 2 has 93, i.e. v85b request 2 matches v78 request 1 there. So the request-2 variation is a state effect that exists without the new paths (v78 itself differs between its two requests), and the one-line disagreement between the v85b and v78 request-2 logs is consistent with that state-dependent variation, not evidence of a defect in the batched out-projection or GPU indexer. Request 1, which does not have that state effect, is bit-identical to v78 over all 2,520 lines. Still unproven without a same-binary request-2 repeat; keep it on the list.

Cause of the v85c/d/f request cancellations (HTTP 408 "generation cancelled") -- CORRECTED: it was this session's own wait loop, not another session. The loop's give-up test used the shell variable `$SECONDS`, which in a long-lived tool shell is already large, so the loop ended ~50 s into the run and its next step (`kill <server pid>`) sent SIGTERM mid-request; the server logged `[Shutdown requested ...]` and cancelled the generation, and the client saw 408. The earlier note blaming another session's `pkill` was wrong. Use `date +%s` deadlines in wait loops.


## v86-v90: request-2/3 route differences traced to a stale GPU mirror (fixed), new best v89 (2026-11-07)

- **Root cause.** The device mirrors for GPU head attention (window and compressed keys, v70+) and the new GPU indexer keys were reset only inside `dsv41_attention_impl` at `layer 0 && pos_base == 0`. The full-graph prefill that actually runs resets the host compressed state in `ggml_deepseek_v41_graph.cpp` and never reached that branch (debug counter: `inval_calls=0`, mirror `ik_synced` still 1022 in request 2). So request 2 and later used the previous request's compressed/indexer keys for blocks the new request generates differently (mismatches started at the first decode-generated block, position 964, in request 2 only). This also affected the v70-v78 head-attention mirror: v78's request 2/3 used stale compressed keys; its request 1 is correct.
- **Fix (commit c114120).** `dsv41_head_attention_invalidate()` is now also called where the full-graph prefill resets the compressed state.
- **Evidence.** (a) v86 (GPU indexer off, batched out-projection on, old mirror reset) is bit-identical to v78 for all three requests, so the batched out-projection is exact. (b) v85g/v87/v88 (GPU indexer on, stale mirror) differ from v78 only in requests 2 and 3. (c) v89 (fix, GPU indexer on) vs v90 (fix, GPU indexer off via temporary edit, reverted): 0, 0, 0 differing decode route lines across three requests, so the GPU indexer is bit-identical to the CPU loop including after the first request. (d) v89 request 1 is bit-identical to v78 (0 of 2,520). v89 requests 2 and 3 differ from v78 (396 lines) because v78's own requests 2/3 were affected by the stale head-attention mirror. Request 2 vs 3 inside v89 differ by 22 lines (residual state variation not explained; also present without the GPU indexer, v90).
- **v89 timing (3 requests, 959+64, host quiet, models on NVMe, direct_io=true, startup margin 3.0 GiB):** decode 12.59 / 12.75 / 12.70 tok/s (p50 78.1 / 76.6 / 76.7 ms), prefill 70.46 / 71.61 / 71.66 tok/s; attention total 28.5 ms/token (context_node 14.75, outproj_node 7.44). Same fix without the GPU indexer (v90): decode 11.67 / 11.74 / 11.77. `check_full_graph_moe_run.py --full-fit --require-full-graph-prefill` PASS.

## v91: grouped out-projection A+B in one device pass (commit 9636e1a), 2026-11-07

`grouped_output_proj_ab_host`: the [groups*rank] intermediate stays on the GPU between attn_output_a and attn_output_b (one upload, one download, same kernels and order). Run `build_mtp_results/2026-11-07-outproj-ab-v91/` (clean tree, models on NVMe, 3 requests, host quiet):
- Decode 12.86 / 12.96 / 12.92 tok/s (v89: 12.59 / 12.75 / 12.70), p50 76.5 / 75.3 / 75.9 ms; prefill 70.58 / 71.81 / 71.89 tok/s.
- `outproj_node` 7.44 to 6.10 ms/token; attention total 27.3 ms/token (context_node 14.65, other 6.5).
- Decode routes bit-identical to v89 for all three requests (0, 0, 0 differing lines); graph checker PASS.

## llama.cpp teacher-forced validation: not possible at the pinned commit (checked 2026-11-07)

- The main GGUF declares `general.architecture = deepseek41` (plus `deepseek41.engram.*` keys, split.count 6). Pinned llama.cpp `8baabd53` registers `deepseek4` only (`LLM_ARCH_DEEPSEEK4`), has no `deepseek41`, and contains no Engram implementation (the only "engram" text in `src` is a comment in `llama-model-loader.h`). The Engram sidecar `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` is a 2-tensor file with architecture `deepseek41-engram` and no split metadata, so llama.cpp could not load it as shard 6 either.
- So `scripts/validate_vs_llama.sh` cannot produce a reference for this model at the pinned commit. Options: a llama.cpp revision that supports `deepseek41` + Engram (none known here), or an in-project reference: compare each rounding-changing GPU kernel against the project's CPU reference (as `tests/test_dsv41_grouped_outproj.cpp` does for the grouped out-projection) and track decode route/logit drift against a CPU-only ZeroMoE run with the same tokens.
