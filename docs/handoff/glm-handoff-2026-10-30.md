# GLM-5.3-Flash GSQ-RCO: session handoff (2026-10-30)

Companion documents: `glm-test-summary-2026-10-30.md` (every test, numbers, verdicts), `glm-gsqrco-performance-handoff.md` (running record with the Nsight timeline and the
earlier investigation), `../models/glm53-flash.md`, `../../configs/bench/perf-baselines.md` (performance rows with model file, quant, size, bpw and version hash).

## 1. State in one paragraph

GLM-5.3-Flash GSQ-RCO decodes at **14.1-14.5 tok/s on the 3090 + 5060 Ti** (69-71 ms/token, clean runs) with zero SSD misses; the goal of ~20 tok/s is **not reached**. Master (`48ddafa`
plus this handoff) holds the shipped settings, the full-graph prefill default for GLM, a registration-race fix, a prefill planner margin and all tooling. The 5060 Ti overlap was implemented
and is numerically exact but gave no measurable speedup (-0.7 ms), so it is **not** on master (branch `glm-gpu1-concurrent`). Long prompts work but prefill slowly (5.7 tok/s at 2,313 tokens).
No valid 3090-only speed exists after the memory change.

## 2. What was done

1. Recovered the ~14 tok/s: pool 88 GiB (zero SSD misses), 10 decode threads, 14.2 GiB hot experts on the 5060 Ti (13.0 -> 14.2 tok/s from threads alone; 5.3 -> 10.7 tok/s on the 3090 alone from pool 80 -> 88).
2. Built a repeatable test setup: `bench/glm_gsqrco_pool.sh` (quiet-box gate, memory watchdog, contamination report, CUDA driver log, abort backtrace) and the config `configs/bench/glm53flash_gsqrco_offload_pool.toml`.
3. Measured and closed levers with clean A/Bs (GPU assist, Q2_K VNNI kernel, more resident experts, adaptive promotion, dense-host-copy tricks, AJ comparison).
4. Characterised the decode with an Nsight node trace: CPU cold experts 54% of a token with both GPUs idle; the 5060 Ti branch sits on the critical path (211 us/layer).
5. Found that the layered prefill is not reproducible run to run and that the full-graph prefill is bitwise reproducible, which makes bitwise parity tests possible.
6. Found and fixed three bugs (below), made full-graph prefill the GLM default like the other families, removed `ZEROMOE_GLM5NEXT_FULL_GRAPH_PREFILL`.
7. Retried the 5060 Ti overlap under the bitwise parity setup (result: exact, no gain).
8. Recorded everything (perf doc with model file/quant/bpw/hash per record; docs; this handoff).

## 3. What was found (the points that change decisions)

- **Time per token** (3090 + 5060 Ti): CPU waits for the 3090 20%, CPU cold experts 54%, GPU combine + 5060 Ti branch + token tail 27%. 3090 kernels are busy 25% of the token, 5060 Ti 3%. The GPU "imbalance" is not the limit; both GPUs idle while the CPU works.
- **DRAM bound**: ~1.8 GB of cold experts per token against ~60 GB/s of DRAM read; the CPU part cannot go below ~30 ms, so ~61 ms/token (~16 tok/s) is a floor unless fewer bytes are read per token.
- **Ranking mismatch**: the shipped usage ranking was collected on the AJ file; the GSQ-RCO file routes differently (77% top-1 agreement). A ranking from 6 pilot prompts cuts cold routes 54% -> 41% and gives 15.84 tok/s **on those same prompts** (in-sample).
- **Q2_K kernels**: a wider/VNNI Q2_K dot is bit-exact and 1.11x in a microbenchmark but 1.9 ms slower in the engine: the lane is bound by memory access and thread sync, not dot instruction count.
- **Determinism**: layered prefill differs from itself (cosine 0.92-0.99); full-graph prefill is bitwise identical. The earlier overlap attempt's "parity failure" was layered-prefill noise.
- **Host memory**: `MemAvailable` minimum 29-31 GiB with the 5060 Ti holding experts, ~20 GiB on the 3090 alone at pool 84 (my earlier 7 GiB estimate was wrong). The host is shared: compile storms from other agents inflated decode 3-9x several times.

## 4. What was fixed (all on master)

| Fix | Commit | Where |
|---|---|---|
| removed the dense host-copy release (it zeroed attention/logits for every request after the first) | `b7d6bd1` | `models/glm/*` |
| RAM-tier slab registration race ("Copy is larger than memobj size", the intermittent `invalid argument` aborts) | `484520b` | `core/engine.hpp` (`slab_pin_target`) |
| prefill planner margin 0.15 -> 1 GiB for GLM5Next (long-prompt GPU out of memory) | `618d454` | `models/glm/ggml_glm5next_graph.cpp` |
| full-graph prefill default for GLM, env var removed | `28114c5` | `models/graph/full_graph_models.cpp`, bench scripts |
| 10 decode threads default, prefill min(threads, 6) (kept by the other agent's cleanup: `--threads` / `omp_max_threads`) | `ce4c313` / master | `ggml_glm5next_graph_decode.cpp`, `ggml_glm5next_graph.cpp` |
| tooling: launcher, tracer, compare tools, Nsight scripts | various | `bench/` |

## 5. Next levers (ordered)

1. **Make the 5060 Ti overlap actually overlap** (est. ~7 ms/token, to ~15.6 tok/s). The code change (`4eb6825`, branch `glm-gpu1-concurrent`: `hooks.build_gpu_overlap` expands the branch before the CPU marker) is bitwise-exact but gave no gain. **Not diagnosed.**
   Hypothesis: the scheduler runs splits one at a time in node order, and the CPU lane's work starts only when the scheduler reaches the CPU marker, so the 5060 Ti split (with its host-staged hops, no P2P) still completes first. First step: an Nsight node trace of binary B
   (template `build_glm_results/2026-10-30-b6d190-glm-node-trace/retry/profile.sh`, analysis `bench/glm_nsys_lane_breakdown.py`) to see whether the 5060 Ti kernels now sit inside the CPU lane range. Fix candidates: start the CPU lane from the fork on its own thread, launch the 5060 Ti split asynchronously without a sync before the marker, return one reduced vector from the 5060 Ti instead of eight (128 KB -> 16 KB hop).
2. **Fewer cold experts per token** (the only lever that can reach 20): a ranking that generalises. The in-sample test shows the mechanism (+11%); a held-out survey is needed to size the real gain (a survey was ruled out by decision). Tools: `bench/make_survey_sets.py`, `graph_usage_record_file`, `bench/make_mix_ranking.py`.
3. **Long-prompt prefill speed** (2,313 tokens in 408 s): `graph_prefill_op_offload = true` runs the 3090 out of memory; needs a planner that accounts for it, or a smaller 3090 expert cache for long prompts. 13-104 token prompts are fine (20-31 tok/s).
4. **3090-only measurement at pool 88** on the merged tree (memory should leave ~15 GiB); use the retry-until-clean pattern.
5. **Small items**: 3090 post-lane kernel chain (~110 small kernels/layer) and token tail, 1-2 ms each.
6. **Speculative decoding**: no draft source for this GGUF (no `nextn` tensors, no small GLM model).
7. **Checks owed**: Qwen family regression run on the slab-pin fix; planner margin on other GLM files/GPUs; merged tree on 3090 only.

Closed by measurement (do not repeat without a new idea): GPU assist, Q2_K AVX512-VNNI dot, more resident experts, adaptive promotion in pool mode (graph-owned slots need a source; old replay: not worth it), dense host-copy release/deferral, pool pinning off, deferred-unregister flush.

## 6. Branches and worktrees

| Branch | State |
|---|---|
| `master` | `48ddafa` + this handoff; everything above except the overlap |
| `glm-gpu1-concurrent` (`4eb6825`) | the 5060 Ti overlap experiment; exact, no gain; not merged |
| `glm-merge-final`, `glm-gpu1-overlap`, `glm-ranking-refit`, `glm-docs`, `glm-land`, `glm-slab-pin-fix`, `glm-memory-bar`, `glm-lane-diag` | merged or superseded history (`glm-lane-diag` holds the old per-layer finite-check diagnostics: do not merge) |
| `glm-handoff` | this document |
| `codex/glm-adaptive-promotion-3090` (worktree `ZeroMoE-wt-land`) | **the other agent's**, left untouched (its handoff `glm-flash-performance-session-2026-10-30.md` is untracked there) |

Worktrees: `ZeroMoE-wt-rank` (build tree, branch `glm-gpu1-overlap`, build dir with the ggml-land prebuilt), `ZeroMoE-wt-merge2` (docs/merges). Binaries of the last A/B are saved: `build_glm_results/2026-10-30-overlap/ov2/binA/zeromoe` (master `48ddafa`) and `binB/zeromoe` (`4eb6825`), each with its `zeromoe.build-info.json` (clean stamp).

## 7. Scripts, configs, data

| File | Use |
|---|---|
| `bench/glm_gsqrco_pool.sh TAG [POOL] [prompt-file]` | the launcher. Env: `GLM_MODEL` (required), `CVD` (0 = 3090 only, default 0,1), `GPU1_HOT_GIB` (14.2), `CPUT` (threads), `NGEN`, `ZEROMOE_BIN`, `GLM_RANKING`, `MOE_CACHE_GB`, `GLM_EXTRA_TOML`, `GLM_EXTRA_ENV`, `GLM_EXTRA_ARGS`, `GLM_PRELOAD`. Waits for a quiet box (idle GPUs, no other model run, load < 3, > 100 GiB free), watchdog at 6 GiB, prints steady ms/token and the contamination report. **Never edit it while a run uses it** (bash reads scripts incrementally). |
| `configs/bench/glm53flash_gsqrco_offload_pool.toml` | run config (offload pool 88 GiB, `host_ram_gb = 16`, 3090 `moe_cache_gb = 12.4`, no GPU assist) |
| `bench/cmp_decode_logits.py A B` | per-step bitwise / max abs / rmse / cosine / top-1 for two `ZEROMOE_DECODE_LOGITS_DUMP` files |
| `bench/hostreg_trace.c`, `bench/abort_backtrace.c` | `LD_PRELOAD` tracers (pinned-memory registrations with stacks; backtrace on abort); build with `gcc -shared -fPIC -O1 ... -ldl -lpthread` |
| `bench/glm_nsys_lane_breakdown.py`, `bench/glm_nsys_gpu1_branch.py DIR` | per-token timeline from an Nsight SQLite export |
| `bench/make_survey_sets.py`, `bench/make_mix_ranking.py`, `bench/q2k_dot_bench.cpp` | survey sets, ranking mix, Q2_K dot benchmark |
| `docs/ssd-offload/data/glm53flash_gsqrco_pilot6_usage_decode.txt`, `bench/data_glm_gsqrco_pilot6_prompts.txt` | pilot ranking counts and the 6 prompts |

Reproduce the baseline (3090 + 5060 Ti):
`GLM_MODEL=/data/sda/models/glm53flash/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf NGEN=96 bench/glm_gsqrco_pool.sh OUTDIR/run1 88 bench/data_glm_gsqrco_pilot6_prompts.txt` (about 10 minutes: ~4 min pool seeding, then the prompts).
Parity (bitwise): `NGEN=8 GLM_EXTRA_ARGS=--raw-prompt GLM_EXTRA_ENV="ZEROMOE_TEACHER_FORCE_TOKEN_FILE=<tokens> ZEROMOE_DECODE_LOGITS_DUMP=<out.bin>"` with the prompt "The capital of France is" (tokens in `build_glm_results/2026-10-30-overlap/teacher.tokens`), then `bench/cmp_decode_logits.py`.
Results: `build_glm_results/2026-10-30-overlap/` (determinism `det`, `det2`; planner `plan`; tracer `trace`, `fix`; headroom `vram`, `opoff`; overlap A/B `ov2`), `.../2026-10-30-rank-survey/` (pilot, ranking, headroom, Q2_K A/B `q2kab`, 3090-only `solo`), `.../2026-10-28-150f68a-gsqrco-baseline/` (earlier series), `.../2026-10-30-b6d190-glm-node-trace/` (Nsight trace by the other agent).

## 8. Cautions for the next session

- The host is shared. Gate every benchmark on a quiet box, read the contamination report, discard contaminated runs, never `pkill` by name (kill your own PIDs after checking `/proc/PID/cmdline`), build with `nice -n 19 -j3`.
- Master was reset once by another session (`reset: moving to 613d9fe`) which dropped a merged commit of mine; check `git reflog master` and the ancestry of your commits after merging.
- The build stamp (`tools/build_provenance.py`) needs a clean tree from configure to the end of the build: do not edit tracked files in a build worktree while it builds.
- Quote only clean numbers; label in-sample ranking results as in-sample; test any memory-saving change with a session of at least two prompts.
