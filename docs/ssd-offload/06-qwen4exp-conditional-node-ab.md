# Qwen4Exp: early-ready (no conditional nodes) vs staged Phase-2 (conditional composition), 2026-10-09

Harness `bench/option_b_early_ready_check.py`, binary `build_acc/moestream` (accepted ggml), config
`configs/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml` (Q2_0, 8,421 resident experts), 64 tokens, 10 threads, six
order-balanced pairs, no dumps. Baseline = early-ready + resident tables + async output (no conditional nodes). Candidate =
staged GPU child + storage service + completion graph + master graph (conditional node at layer 13, `PARTIAL_LAYER_PHASE2=1`).

| arm | tok/s per rep | mean |
|---|---|---|
| early-ready | 31.1 33.9 34.0 31.3 33.9 33.7 | 32.99 |
| staged / conditional | 24.7 24.2 24.5 24.2 24.6 24.6 | 24.48 |

-25.8% (paired -20.6 .. -28.5%, 0/6 wins). The earlier -33.7% was measured before the staged payload was copied from the resident
DDR arena instead of `pread` (commit 5aa754a); the remaining gap is structural.

Per-token profile (`MOESTREAM_QWEN4EXP_GRAPH_PHASE_TIMING=1`, steady tokens): early-ready 29.4 ms, staged 40.6 ms (+11.2 ms).
Staged phases (mean of the last 50 tokens): storage request 3.42 ms, payload_prepare 2.35 ms (inside upload_submit 2.39 ms), upload
H2D 1.15 ms, conditional host 2.68 ms + device 1.60 ms, tail replay 22.8 ms (submit 21.7, sync 1.1). Storage + upload run serialized on
the host before the tail graph launches (~5.8 ms); the conditional-node machinery is ~4.3 ms (whether it overlaps the tail replay
is not separated by these timers).
