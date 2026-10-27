# MTP side-task shared notes — 2026-10-27

## Worktree and device sync

- Side-task worktree: `/home/ray/Projects/ZeroMoE-wt-mtp-side-work`, branch `mtp-side-work-active`, HEAD `8d52e70`. This contains the side-task work on top of `6d7d431` and synced active commit `398f129`. The active `wt-swap-b` is now at `9147f07`; the promotion cap 12 is committed in its history (`8d3e8fd`), and I did not modify those feature files. Original `mtp-side-work` at `d2317b2` remains.
- The latest measured ZeroMoE side binary SHA256 `5d8702f5888fb5eee4ebb8d3f655e8a98b3549a0502986d19d3add5828902e6c` was built and provenance-checked at source `6d7d431` (GGML base `8baabd53b`, patch manifest `e4a697188d78d4761be705f7c0fdd7675a8a9cc2ef1c8657117387a9c983ec08`, sm_86). Commit `8d52e70` adds only side-task tooling; no engine source changed after those runs. The benchmark manifests retain the exact tested source/binary identity.
- The active 3090 process is occasionally present as `wt-swap-b/build/zeromoe` using ~276 MiB but idle. Do not overlap GPU runs. Recheck `nvidia-smi` and `pgrep -x zeromoe` before each run. If `nvidia-smi` transiently fails, retry `nvidia-smi -L`, then full `nvidia-smi`.
- `scratchpad/hy/tr/` is absent. The active branch has six tracked prompts in `bench/prompts/mtp_suite/` (btree, leet, math, palindrome, review, story). Use that explicitly as the available six-prompt suite and record that it is a replacement.

## 1. Adaptive-placement accuracy gate

- Added `tools/check_adaptive_placement_accuracy.py`. It captures baseline-generated tokens, forces the same tokens through candidate placement, and reports per-token logit cosine, top-1 agreement, KL, and teacher-token NLL.
- `--help` succeeds. It supports usage-set or top-K variants, explicit prompt directories, and engine provenance from another worktree.
- Completed all six prompts with 256 teacher-forced tokens each, static K=13,000 vs K=11,000. Baseline token stream was forced through the candidate; all six produced exactly 256 logit records.
- Macro mean over the six prompts: **99.09% top-1 agreement**, mean logit cosine **0.999392**, mean KL **0.000592949**. Prompt-level results:

  | Prompt | Top-1 agreement | Mean logit cosine | Mean KL |
  | --- | ---: | ---: | ---: |
  | B-tree | 99.22% | 0.999469 | 0.000576569 |
  | LeetCode | 98.83% | 0.999662 | 0.000568193 |
  | Math | 98.44% | 0.999318 | 0.000405293 |
  | Palindrome | 99.61% | 0.999488 | 0.000414409 |
  | Code review | 99.61% | 0.999006 | 0.000374523 |
  | Story | 98.83% | 0.999408 | 0.001218709 |

- This uses the tracked replacement suite `bench/prompts/mtp_suite`; `scratchpad/hy/tr/` is still absent. The result directory contains per-token comparisons, configs, logs, manifest, and build provenance.

## 2. Bee / llama.cpp Q2_0 MTP reference

- Added `tools/splice_qwen4exp_mtp_gguf.py`. It created three shards under `/home/ray/vllm-workspace/qwen3.8-flash-next/Q2_0/MTP-spliced/`; original model shards and sidecar were preserved.
- The spliced metadata was checked: 1,256 tensors, Qwen4Exp architecture, `block_count=49`, `nextn_predict_layers=1`. Bee's `llama-gguf <shard> r n` returned success for all three shards; this validates containers and metadata, not model-load compatibility.
- Created a corrected spliced GGUF under `/home/ray/vllm-workspace/qwen3.8-flash-next/Q2_0/MTP-spliced-corrected/`. It now replaces `qwen4exp.attention.compress_ratios` with the sidecar's 49-entry array after verifying its first 48 values match the main shards; header data offsets remain unchanged. All three shards pass Bee's `llama-gguf <shard> r n`.
- The existing ready Bee binary at `/home/ray/Projects/beellama.cpp/build-qwen38-rtx3090/bin/llama-speculative-simple` was built 2026-09-13 from commit `43fed6726`; its source checkout now has Qwen4Exp MTP edits dated 2026-10-27. It is stale relative to that source.
- Bee's early CUDA device enumeration returns no devices, but an isolated C++ probe succeeds after a 500 ms delay; the temporary `LD_PRELOAD` startup-delay helper makes the old Bee binary list CUDA0. No Bee source or binary was modified.
- The previous `expected 1256 tensors, got 1253` failure came from the September Bee binary. The current rebuilt binary loads all 1,256 tensors and runs MTP; see the verified smoke and suite results below.
- The fresh `build_bee_ref` build was stopped at 33% after confirming the existing Bee build tree had matching CMake options and only three NVCC commands left to rebuild. The target was rebuilt incrementally at `/home/ray/Projects/beellama.cpp/build-qwen38-rtx3090`; the current binary and its SHA are recorded below.
- After the fresh build, retry model loading first. If the 1256/1253 mismatch persists, revise the shard contents/tensor count based on the loader's MTP tensor contract before running the six prompts.

## 3. Fixed-length plain/MTP benchmark

- Added `tools/bench_qwen4exp_mtp_suite.py` for alternating plain/MTP runs, 256 tokens, three repeats, per-run logs, medians, and `--greedy-fast-path on|off` / `--prefill-overlap on|off` controls.
- The `ZEROMOE_BENCH_IGNORE_STOP=1` fix is in the side worktree; it suppresses EOS/stop termination without disabling MTP. A pre-sync smoke run emitted 256 tokens, but its speed is not a current-base result.
- Completed the six-prompt, three-repeat harness on the current e87076e side binary: Q2_0, RTX 3090, context 512, 10 threads, K=13,000, 3 drafts, exactly 256 emitted tokens, overlap off. MTP `MTP-GRAPH emitted=256` was checked for every run.
- Pooled median decode-wall rate over 18 runs: **44.28 tok/s plain, 49.50 tok/s MTP (1.12x)**. Median MTP steady-speculation rate was 56.43 tok/s, acceptance 74.4%, 3.22 emitted tokens/cycle, and draft graph setup 539.8 ms.
- Per-prompt decode-wall medians (tok/s):

  | Prompt | Plain | MTP | Ratio |
  | --- | ---: | ---: | ---: |
  | B-tree | 45.48 | 49.45 | 1.09x |
  | LeetCode | 48.63 | 53.00 | 1.09x |
  | Math | 29.70 | 30.70 | 1.03x |
  | Palindrome | 49.34 | 52.33 | 1.06x |
  | Code review | 44.19 | 49.55 | 1.12x |
  | Story | 34.08 | 31.15 | 0.91x |

- The speedup varies by prompt; the story prompt was slower under MTP. These values are reproducible benchmark results, not a best-of-N run.
- The artifact initially recorded MTP token-dump count as 0 because the MTP emitter does not use the ordinary generated-token dumper. The per-run summary was normalized to keep `token_dump_count=0` separate and report the validated `MTP-GRAPH emitted=256` as `generated_token_count`.

## 4. MTP driver overhead and setup overlap

- Added deterministic `SpecGreedy` for temperature zero. `ZEROMOE_MTP_GREEDY_FAST_PATH=0` selects the old probability-vector implementation for A/B; default is direct argmax. Repetition penalty preserves the generated-token 64-token window and duplicate occurrences.
- CPU speculation tests passed across stepwise, batched restore/replay, and prefix-commit modes, including one-hot parity with repetition penalty.
- Current-base GPU correctness A/B passed on the palindrome prompt, 64 generated tokens, RTX 3090, context 512, K=13,000, 3 drafts:
  - Old probability-vector path: byte-identical to plain; 67 drafted, 40 accepted (59.7%), 23 cycles; decode wall 1.949 s (32.84 tok/s); driver/sampling bucket 128 ms.
  - Direct argmax: byte-identical to plain and same acceptance; decode wall 1.812 s (35.33 tok/s); driver/sampling bucket 62 ms.
  - This is one short A/B pair, not a throughput benchmark. It shows about 52% less time in the driver/sampling bucket for this prompt.
- Setup instrumentation measured 550 ms for draft setup. Each of two graph builds uploaded about 2.33 GB of weights, including a separate 0.2 GB vocabulary head: about 200 ms weight upload + 46 ms head upload per build. Scheduler allocation was 0.28 ms. Repeated upload/build dominates setup.
- A prefill-overlap trial launched draft allocation after the target startup graph build and before prompt prefill. Prefill completed, but the target's post-prefill decode preparation then failed: only 0.95 GiB remained, below its 1.00 GiB decode-graph admission floor. The request correctly failed rather than falling back. The code now makes overlap explicitly opt-in and adds a 4 GiB free-VRAM guard (3 GiB draft budget + 1 GiB decode headroom); on the 3090's ~3.31 GiB headroom it should skip and use synchronous setup. The guard was rebuilt/restamped at side-task HEAD `6d7d431`; K=13,000 skips safely, while K=11,800 allows overlap and hides 546.71 ms of setup (see the measured overlap section below).
- The earlier story-prompt mismatch and 2.58 tok/s run were on the pre-sync `d2317b2` build and are discarded as current-base evidence.
- The equivalence helper appends the REPL quit input when `--prompt` is plain text, avoiding the earlier empty-input invocation.

## Remaining optional diagnostic

- All requested side-task deliverables have verified artifacts and are summarized below. The only uncompleted follow-up is the verbosity-5 Bee per-position candidate trace; its first attempt failed because another Python process was using the 3090. The current measured clue is clear, but the exact later-position target/draft mismatch remains unproven.
- Retry that trace only if more acceptance root-cause work is requested and the GPU is idle. No changes were made to the active MoE/pin/LRU/early-ready feature files or to Bee source.

## Update — 2026-10-27 17:01 local

- GPU access recovered: `nvidia-smi` shows RTX 3090 and RTX 5060 Ti idle. GPU queries have been intermittent, so recheck immediately before each inference run.
- Side-task tree was synced to active `wt-swap-b` commit `398f129` as `6d7d431`; the uncommitted promotion cap change (6 to 12) remains excluded. The prior ZeroMoE binary is stamped for `e87076e`, so rebuild/restamp before new side-tree performance measurements.
- Reworked the Bee benchmark `--dry-run`: it now emits all 18 command files for six prompts x three repeats without calling `nvidia-smi` or starting a model. This path ran successfully using the old binary only to generate commands; no performance was measured.
- The original fresh Bee build reached 33% after about 30 minutes while compiling many unchanged CUDA templates. It was stopped after checking the existing `build-qwen38-rtx3090` cache against the fresh cache (40 relevant CUDA/build options matched; dry-run planned three NVCC commands). Current build session `72904` is rebuilding `llama-speculative-simple` incrementally in that existing tree from the same dirty Bee source. Do not benchmark until it exits.
- After the incremental binary links, load-test the corrected spliced GGUF first. Only then run the six-prompt Bee MTP benchmark. Continue with the paired ZeroMoE greedy-fast-path matrix once host compile load is gone.

## Bee loader smoke — 2026-10-27 17:05 local

- The incremental build completed successfully in `/home/ray/Projects/beellama.cpp/build-qwen38-rtx3090`; binary SHA256 `94b663a5515608afd27d877d9c8da15e7f8ccfaec1dc857566872d1c1f4c7241`. Bee source HEAD is `43fed6726` with 19 pre-existing dirty paths; I did not edit Bee source or submodules.
- `--list-devices` with the startup-delay preload found `CUDA0: NVIDIA GeForce RTX 3090` (23,858 MiB). The corrected spliced GGUF now loads 1,256 tensors; the previous 1,256/1,253 loader failure is resolved in this rebuilt binary. The load log reports all 50/50 layers offloaded and CUDA model buffer 17,165.04 MiB, with 5,878 MiB free after load.
- Short MTP smoke (16 requested tokens, 3 drafts) exited successfully and logged 18 generated tokens in 2.354 s; 9/9 drafts were accepted in that one window. This confirms load and MTP execution only; it is too short and includes setup, so it is not a decode-speed result.
- NVML remains intermittent: `nvidia-smi` failed during/after the smoke, then recovered. `fuser` showed an active `wt-swap-b/build/zeromoe` under `nsys profile` on the 3090. Do not start the six-prompt benchmark until that process releases the device.
- The Bee harness `--dry-run` fix was verified by writing all 18 prompt/repeat commands without starting a GPU process.

## Bee exact-length correction — 2026-10-27 17:06 local

- The first Bee suite attempt with CLI `-n 256` stopped the harness because Bee emitted 258 tokens. Inspection of `examples/speculative-simple/speculative-simple.cpp` shows it checks `n_predict > params.n_predict` only after a complete verification window.
- A B-tree calibration with CLI `-n 254` emitted exactly 256 tokens (33.084 s decode, 7.738 tok/s; 128 MTP calls, 128/128 drafts accepted). This is a calibration run, not suite performance evidence.
- The Bee harness now has `--n-predict-offset` (default 1), records both requested output count and CLI `-n` in the manifest, and still rejects any run whose actual decoded count differs. `--dry-run` confirmed 18 commands use `-n 255` for 256 actual tokens.
- Exact-length six-prompt/three-repeat suite is running with `-ncmoe 30`, output directory `bee-mtp-q2-256-exact-r3`. Treat these as a separate 30-CPU-MoE-layer recipe. The user's earlier 22/26 split is the fair comparison target; after this run, test a short 22-layer CPU split load and use it for the comparable suite if it fits.

## Bee MTP reference results — 2026-10-27 17:35 local

- Completed two exact-length six-prompt suites, each with 256 generated tokens and three runs per prompt, current Bee source `43fed6726`, rebuilt binary SHA256 `94b663a5515608afd27d877d9c8da15e7f8ccfaec1dc857566872d1c1f4c7241`, RTX 3090, Q2_0, context 1536, 10 threads, three MTP draft tokens. The runner passes `-n 255` and validates 256 emitted tokens because Bee checks the output limit only after a verification window.
- 22 CPU / 26 GPU MoE-layer split: pooled median decode 9.099 tok/s (per-prompt medians 9.08–9.12), 28.1365 s for 256 tokens. The model buffer is 22,565 MiB; measured peak GPU use was about 23,622 MiB, leaving ~954 MiB. It fits but has little headroom.
- 30 CPU / 20 GPU split: pooled median 7.743 tok/s (per-prompt medians 7.681–7.777), 33.061 s for 256 tokens.
- Corrected the acceptance interpretation in the harness and recomputed both summaries from their raw logs. The old `100%` value counts accepted windows. Candidate-token acceptance is 128/383 = **33.42%**; window acceptance is 100%. Mean output is **2 tokens per verification window**. Both metrics are now explicit in result/summary JSON.
- This is much slower than the user-provided 28 tok/s plain 22/26 baseline, but a same-prompt Bee plain matrix has not yet been run, so that ratio is not yet a matched measurement. Need run Bee plain on the same six prompts and 22/26 split after the GPU is free.
- Current GPU owner at the latest check is another `wt-swap-b` Nsight run (`nsys` wrapping `build/zeromoe`). Do not overlap the plain baseline or ZeroMoE driver A/B.

## Plain Bee comparison runner — 2026-10-27 17:40 local

- Added `tools/bench_bee_plain_reference.py` in the isolated side-task tree. It records source/build/model/prompt identity and decode-only `llama_print_timings eval time`, and pairs with the MTP suite at Q2_0, 22/26 split, context 1536, 10 threads, and 256 tokens. It checks GPU ownership via NVML and falls back to `fuser` when NVML fails; command output streams to raw logs instead of accumulating in memory.
- Python compilation and a one-prompt dry-run passed; it generated a plain `llama-cli` command with `-n 256`, `-ncmoe 22`, and no speculative flags.
- The 8-token plain smoke was stopped before decoding after ~2 minutes and ~34 GB read from the cold main GGUF. It produced no timing result. This is consistent with a cold model load, but that explanation is not yet verified. The command used `llama-cli -n 8`; next run should stream its log and wait for model loading to finish.
- At the latest check, `wt-swap-b`'s Nsight-wrapped ZeroMoE run had reclaimed the 3090. Do not run plain benchmark or side-tree MTP A/B until `fuser` and `nvidia-smi` show it idle.

## Matched Bee plain/MTP comparison — 2026-10-27 17:45 local

- Plain Bee first pass: six prompts, one run each, exact 256 output tokens, Q2_0 main shards, RTX 3090, 22 CPU / 26 GPU MoE-layer split, context 1536, 10 threads. Decode-only timing uses `llama_print_timings eval time`.
- Plain rates: B-tree 26.31, LeetCode 24.47, Math 27.29, Palindrome 25.72, Review 27.31, Story 26.61 tok/s; median **26.46 tok/s**, range 24.47–27.31.
- Matched per-prompt comparison against the 3-run MTP medians: 9.09 vs 26.31 (B-tree), 9.09 vs 24.47 (LeetCode), 9.08 vs 27.29 (Math), 9.12 vs 25.72 (Palindrome), 9.10 vs 27.31 (Review), 9.11 vs 26.61 (Story). Median MTP/plain ratio **0.344x**. Plain is roughly 2.9x faster in these runs.
- The plain matrix was intentionally stopped after one six-prompt pass because MTP was much slower. The runner entered repetition two before the stop took effect; three repetition-two runs completed/started and are excluded. The accepted comparison uses only the six `rep-01` records. It is directional, not a repeated plain median.
- This directly verifies the gap against a same-prompt baseline: the user's earlier 28 tok/s plain figure is close to this 26.46 first-pass median. Bee MTP is slow despite 100% window acceptance because candidate-token acceptance is only 33.42% and output is 2 tokens/window. Likely target-window CPU expert work dominates; verify with per-window target/draft timing before treating that attribution as proven.

## ZeroMoE greedy fast-path A/B — 2026-10-27 (current side binary)

- Rebuilt/restamped side-task source HEAD `6d7d43195b47188f4602c03c76ee4e9e99f9ac59`; binary SHA256 `5d8702f5888fb5eee4ebb8d3f655e8a98b3549a0502986d19d3add5828902e6c`; GGML base `8baabd53b`, patch manifest `e4a697188d78d4761be705f7c0fdd7675a8a9cc2ef1c8657117387a9c983ec08`.
- Ran the same six prompts once each per mode: Q2_0, RTX 3090, K=13,000, context 512, 10 threads, three drafts, exactly 256 emitted tokens, prefill overlap off. Raw logs/configs/manifests are under `build_mtp_results/mtp-greedy-fastpath-on-r1-6d7/` and `mtp-greedy-fastpath-off-r1-6d7/`.
- Median driver/sampling time fell from **426 ms** on the dense probability-vector path to **214 ms** on direct argmax, a **49.8%** reduction across these prompts. Per-prompt reductions were 48.9–50.8%.
- Median MTP decode-wall rate was **48.72 tok/s on** vs **46.88 tok/s off**; median plain rates were 45.67 and 45.30. This is one run per prompt, so the ~3.9% MTP wall gain is directional; the profile-bucket reduction is consistent across all six.
- Story remained below plain (29.66 tok/s on, 30.16 off), and the one-run sample is not a repeatable performance claim. Prior 64-token A/B was byte-identical to plain; CPU speculation tests cover all modes. The full six-prompt results were recorded with the current 6d7 binary.

## llama.cpp MTP setup and acceptance follow-up — 2026-10-27

- Upstream's `examples/speculative-simple/README.md` shows `draft-mtp`, `--spec-draft-n-max 3`, `-ngld 99`, `-ngl 99`, greedy `--temp 0 --top-k 1 --seed 42`, fixed `-n 256`, and `-lv 4`. Our Bee run matches these core settings; the 22 CPU / 26 GPU MoE-layer split and Q2_0 target are hardware-specific. The README is an example, not a guarantee that n_max=3 is optimal for every quant/split. [Upstream README](https://github.com/ggml-org/llama.cpp/blob/master/examples/speculative-simple/README.md)
- The official Qwen3.8 MTP issue reports a different Q3_K_XL / 4x P40 setup and sidecar with 0.446 acceptance, mean accepted length 2.33, and 30.6 tok/s prose vs 21.6 tok/s plain after a loader fix. It is not a direct hardware/quant comparison, but confirms a working Qwen3.8 MTP path can improve throughput. [Issue #29174](https://github.com/ggml-org/llama.cpp/issues/29174)
- The Bee runner's 22/26, K=13,000-equivalent layer split, Q2_0, greedy settings, draft model on GPU, and three-token max were all explicit and validated. Its 100% figure is accepted windows; the raw stats are 128/383 candidate tokens accepted and `#acc rate/pos=(1.000)` with only one position listed. Thus position 1 is accepted every window while positions 2 and 3 add no accepted tokens. This locates the low acceptance in later draft positions; the cause is still unproven.
- Tested the official issue's `--spec-draft-p-min 0.75` style cutoff over six prompts, one run each, same Q2_0/22-26 setup. All six were unchanged: 9.08–9.10 tok/s, 33.42% candidate-token acceptance, 2.00 tokens/window. Logs confirm `p_min=0.75`; it did not shorten drafts.
- Tested `n_max=1` on the palindrome prompt. It generated 128 candidates, accepted all 128, and emitted 2 tokens/window at 11.262 tok/s vs n_max=3's 9.12 tok/s on the corresponding three-run median; plain was 25.72 tok/s on its one-run first pass. This is a diagnostic single prompt, not a general performance claim.
- The target log reports 783 eval tokens for n_max=3 versus 528 for n_max=1 on the same ~271-token prompt. Subtracting the prompt leaves about 512 vs 256 verification rows for 128 windows; the two extra candidates roughly double target verification rows and cost about 5.16 s. This aligns with the speed gain from n_max=1. It suggests verification work is the main cost; a per-position target-forced trace is still needed to explain why positions 2 and 3 are wrong.

## Confidence-cutoff follow-up — 2026-10-27

- Added and verified `--draft-p-min` in the Bee harness. The p_min=0.75 six-prompt run and p_min=0.99 palindrome run both kept `n_max=3`, generated 383 candidate tokens over 128 windows, accepted 128, and emitted 2 tokens/window. The p_min=0.99 palindrome speed was 9.152 tok/s vs 9.12 tok/s n_max=3 median; no meaningful change.
- Local Bee source initializes the MTP draft sampler with top-k 10 and checks `p_min` against the sampler's top candidate after filtering. The cutoff did not reduce candidate count even at 0.99, so this setting does not address the high-confidence later candidates rejected by the target on this run.
- Palindrome n_max=1 emitted exactly 256 tokens at 11.262 tok/s, with 128/128 one-token drafts accepted and mean window length 2.0; n_max=3 was 9.12 tok/s on its three-run median. This isolated run supports the hypothesis that evaluating later rejected candidates adds substantial target work, but n_max=1 still trails the 25.72 tok/s plain first-pass result.
- The first useful clue is therefore the draft rollout: position 1 agrees with the target every cycle, positions 2+ do not. The exact model/hidden-state reason remains open. Compare forced target logits and draft IDs by position before attributing it to quantization or a model limitation.

## Successful setup-overlap trial — 2026-10-27

- At K=12,000 the guard saw 3.99 GiB free and correctly skipped overlap; setup stayed synchronous at 547.83 ms, hidden 0 ms.
- At separately recorded K=11,800, the 4 GiB guard passed: `[MTP-GRAPH-PREFETCH] started after target graph admission`, setup completed during prefill, `hidden_ms=546.71`, `post_prefill_wait_ms=0.01`. Total setup was 546.72 ms; almost all of it was hidden by prefill.
- One review prompt, 256 tokens, RTX 3090, context512, 10 threads, 3 drafts: MTP 51.44 tok/s wall vs plain 41.85 at K=11,800. This is a one-run setup test and a different K from the K=13,000 baseline; treat the speed comparison as directional. Raw logs/configs are in `build_mtp_results/mtp-prefill-overlap-k11800-p075/`.
- This completes the setup-overlap subtask. K=13,000 remains guarded and uses synchronous setup because it lacks the 4 GiB admission headroom.

## Per-position trace attempt — 2026-10-27

- Source inspection confirms Bee MTP logs draft candidates only at verbosity 5 (`SPC_DBG`: position, token ID, top-k probability and piece). The harness now exposes `--verbosity` (default 4); a one-prompt `-lv 5` run was attempted without changing Bee source.
- The attempt failed during target context allocation (`cudaMalloc` for a 392 MiB compute buffer). After exit, `nvidia-smi` showed 818 MiB already in use and 44% utilization, with a Python process holding `/dev/nvidia0`; this was external GPU contention. No candidate trace or speed result was produced. Retry only when `nvidia-smi` and `fuser` show the 3090 idle.
- Code inspection also shows the draft sampler applies top-k 10 before `p_min` checks `cur_p->data[0].p`. The p_min 0.75 and 0.99 runs both generated all 383 candidates; a likely explanation is that the confidence is normalized after top-k and remains above threshold. This explains the cutoff no-op, not why later candidates disagree with the target.

## Commit status — 2026-10-27

- Side-task implementation and tools are committed on `mtp-side-work-active` as `8d52e70` (`MTP side tasks: accuracy gate, Bee reference, benchmarks`). The `ZeroMoE-wt-swap-b` working tree is not part of that commit.
- This shared note is being committed separately on `swap-b`; the active feature branch's files are not staged.
