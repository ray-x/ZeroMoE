# Qwen4Exp decode: why some runs land at 31-32 tok/s instead of ~34 (2026-10-08)

## Finding
Slow decode runs occur only while **nvtop** is running on the box. With nvtop stopped, 48 consecutive
recipe runs (two batches) ranged 33.17-34.17 tok/s (mean 33.79-33.80, sd 0.18-0.20) with none below 33.0.
With nvtop and htop running, 24 runs ranged 31.83-33.93 (mean 33.26, sd 0.62) and 7 were below 33.0.
htop was not the cause (see caveat). The mechanism inside nvtop (NVML polling of the GPU vs CPU use) is
NOT proven: the planned synthetic-poller control batch was not run.

## Setup (all runs)
- Build: copy of the cleanup/ggml-exl3-only working tree linked to the accepted ggml
  (llama.cpp-moestream-integration/ggml), HEAD 11c6100 content; Q2_0 split-44 config, 28-token prompt,
  64 teacher-forced tokens, `MOESTREAM_QWEN_CPU_THREADS=10`, early-ready recipe on (ALL_SPLIT, RESIDENT_TABLES,
  ASYNC_OUTPUT = 1), both bench arms identical, 12 alternating pairs per batch. GPU guard clean in every batch.
- Host: Xeon W-2235 (6 cores/12 threads, governor powersave via intel_pstate), RTX 3090, THP=always.

## Results (decode tok/s, steady steps)
| batch | monitors | n | min | median | max | mean | sd | runs < 33.0 |
|---|---|---|---|---|---|---|---|---|
| earlier (lowrun2) | nvtop + htop running | 24 | 31.83 | 33.39 | 33.93 | 33.26 | 0.62 | 7 |
| A | nvtop + htop SIGSTOPped | 24 | 33.48 | 33.83 | 34.17 | 33.80 | 0.18 | 0 |
| C | nvtop stopped, htop running at start | 24 | 33.17 | 33.83 | 34.06 | 33.79 | 0.20 | 0 |

## Ruled out
- GPU clocks/P-state/throttle: SM 1893-1965 MHz in all runs, mem fixed, P2, no throttle reasons.
- CPU frequency: top-10 cores 3.92-3.99 GHz in all runs; slow runs not lower.
- Temperature: rises with time, unrelated to speed.
- Transparent huge pages: AnonHugePages is 94.1% of RSS (34.6 of 36.8 GiB) in every one of 24 runs.
- Per-token stalls: slow runs are slower at every percentile (+7-10% per step, small constant part).
- Engine-reported statistics (memory, paging, disk read) do not differ between slow and fast runs.
- Thread placement is a symptom, not the cause: in the nvtop-running batch speed correlated -0.92 with the
  number of physical cores hosting two of the 10 busiest threads (fast runs: exactly 4.00 pairs, the minimum
  for 10 threads on 6 cores; slowest: 4.8-4.9). With monitors paused the correlation vanished (-0.07) and
  runs with poor placement were still fast.
- Other CPU users in the decode window (my own Claude process 5-12 ticks, codex, kernel workers) were small
  and did not explain the tail (corr with other-process ticks -0.59 before, -0.23 after pausing).

## Caveats
- Batch C: htop was running (S) when the batch started, but was found stopped afterwards, so it may not
  have run for the whole batch. htop is therefore "probably innocent", not proven.
- Only one prompt and one 64-token continuation. Sampling was 5-10 Hz over ~1.8 s decode windows.
- The sampler's own overhead (and the guard, bench and my agent process) were present in every batch.

## Recommendation
- Do not run nvtop (or other GPU pollers) during decode benchmarks; expect ~33.8 tok/s (sd 0.2) without it.
- Suggested hardening, not done: add `nvtop` to the process names the bench treats as contention
  (competing_cpu() in bench/option_b_early_ready_check.py) so a run aborts instead of silently coming out slow.
- Optional confirmation: run a batch with nvtop stopped and a synthetic `nvidia-smi -lms 1000` poller; a
  returning tail would point at GPU polling as the mechanism.

Raw data (kept outside the repo, /home/ray/work/moestream-root-snapshot/logs): lowrun2_20261008*, expA_paused_unpinned_20261008*,
expC_htop_only_20261008*; analysis: analyze_lowrun2.py, sampler2.py / sampler3.py.
