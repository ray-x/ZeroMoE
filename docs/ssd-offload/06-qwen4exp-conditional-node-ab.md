# Qwen4Exp: early-ready (no conditional nodes) vs staged Phase-2 (conditional composition), 2026-10-09

Harness `bench/option_b_early_ready_check.py`, binary `build_acc/zeromoe` (accepted ggml), config
`configs/testing/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml` (Q2_0, 8,421 resident experts), 64 tokens, 10 threads, six
order-balanced pairs, no dumps. Baseline = early-ready + resident tables + async output (no conditional nodes). Candidate =
staged GPU child + storage service + completion graph + master graph (conditional node at layer 13, `PARTIAL_LAYER_PHASE2=1`).

| arm | tok/s per rep | mean |
|---|---|---|
| early-ready | 31.1 33.9 34.0 31.3 33.9 33.7 | 32.99 |
| staged / conditional | 24.7 24.2 24.5 24.2 24.6 24.6 | 24.48 |

-25.8% (paired -20.6 .. -28.5%, 0/6 wins). The earlier -33.7% was measured before the staged payload was copied from the resident
DDR arena instead of `pread` (commit 5aa754a); the remaining gap is structural.

Per-token profile (`ZEROMOE_QWEN4EXP_GRAPH_PHASE_TIMING=1`, steady tokens): early-ready 29.4 ms, staged 40.6 ms (+11.2 ms).
Staged phases (mean of the last 50 tokens): storage request 3.42 ms, payload_prepare 2.35 ms (inside upload_submit 2.39 ms), upload
H2D 1.15 ms, conditional host 2.68 ms + device 1.60 ms, tail replay 22.8 ms (submit 21.7, sync 1.1). Storage + upload run serialized on
the host before the tail graph launches (~5.8 ms); the conditional-node machinery is ~4.3 ms (whether it overlaps the tail replay
is not separated by these timers).

## No-op staged control (is the cost the conditional node, or the staging?)

The conditional node itself needs no H2D: it is a one-thread device kernel that reads the routed ids and calls
`cudaGraphSetConditional` (`cuda/qwen4exp_master_graph.cu`). The H2D in the profile above is the staged expert payload the
staged child consumes (storage 3.4 ms + payload prepare 2.4 ms + H2D 1.2 ms), which happens before the node.

Control arm: same candidate environment but `ZEROMOE_QWEN4EXP_PARTIAL_LAYER_TOP_N=512` (all 512 experts of layer 13 resident),
so every token takes the `all_resident` dispatch (63/63 `resident-gpu`, no storage request, no payload, no H2D, staged branch
never fires). Proof gates relaxed with `QWEN_AB_SKIP_STAGED_PROOF=1`; six order-balanced pairs:

| arm | tok/s per rep | mean |
|---|---|---|
| early-ready | 31.4 33.7 32.4 32.6 32.0 34.1 | 32.70 |
| two-graph / conditional composition, staged branch a no-op | 34.0 33.8 33.4 34.0 33.8 33.1 | 33.69 (+3.0%, 5/6, within the baseline's own spread) |

So the two-graph split, route readback and resident child are free; the 25.8% loss comes from running the staged branch
(storage + payload + H2D + three-way conditional launch), not from the graph structure. Caveat: with all experts resident the
dispatcher replays the resident child directly and never reaches `try_launch_three_way`, so this control does not execute the
conditional node; the node's own ~4.3 ms (host 2.7 + device 1.6) is only bounded by the staged-path profile. Isolating it needs a
variant that launches the three-way graph with an empty payload (skip storage/upload, keep the launch), which has to keep the
completion-graph ready event consistent to avoid a hang.

## CPU branch instead of H2D staging (forced-CPU arm)

CUDA forbids host nodes inside a conditional body (`cudaGraphInstantiate` -> invalid argument with 12 host nodes in the child;
see memory note `cuda-conditional-node-host-nodes-blocker`), so a CPU branch cannot live *inside* the node. The Phase-2
dispatcher already has the equivalent as a host decision: `ZEROMOE_QWEN4EXP_STAGED_FORCE_CPU=1` skips the storage request,
payload, H2D and both conditional children and runs the CPU suffix graph (63/63 tokens `dispatch=cpu`, output valid).

| arm (six order-balanced pairs) | tok/s per rep | mean |
|---|---|---|
| early-ready | 34.2 33.5 32.2 34.4 33.3 34.1 | 33.62 |
| Phase-2 structure, CPU suffix (no H2D, no conditional children) | 30.3 30.2 30.5 30.4 30.3 30.1 | 30.29 (-9.9%, 0/6) |
| Phase-2 structure, staged-GPU (measured above) | 24.7 24.2 24.5 24.2 24.6 24.6 | 24.48 (-25.8%, 0/6) |

Replacing the H2D staging by CPU compute recovers most of the loss (-25.8% -> -9.9%) but stays 3.3 ms/token behind early-ready: the
CPU suffix starts only after the route readback instead of running concurrently with queued GPU work.
