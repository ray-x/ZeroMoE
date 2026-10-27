# Runtime settings: host-node mode, CPU threads, CUDA schedule

Three environment variables control how the Qwen4Exp CPU/GPU decode path executes. They interact, so they are
documented together. All numbers below were measured on the project's reference box (Xeon W-2235, 6 cores / 12 threads,
RTX 3090, Qwen3.8-Flash-Next UD-IQ1_M, plain 22 GPU / 26 CPU layers unless stated), 80-token decode, idle machine,
interleaved repetitions. Methodology and raw results: `tests/option_b_split_reverify_howto.md` sections 14-16.

## `ZEROMOE_QWEN4EXP_HOST_NODE`

| | |
|---|---|
| Values | `1` enables. Unset or anything else = off (only the first character `1` counts). |
| Default | off |
| Scope | Qwen4Exp full-graph **decode** only (`S == 1`). Prefill is unaffected. |
| Needs | Qwen4Exp full-graph decode (the default); TOML `qwen4exp_cpu_offload = true` and `qwen4exp_cpu_offload_layers = [...]`; complete preloaded DDR expert arenas for those layers (`ram_preload_all_experts = true`). |

**What it does.** Each layer listed in `qwen4exp_cpu_offload_layers` has its routed-MoE FFN replaced by a `GGML_OP_CPY`
marker. While the CUDA graph is captured, the marker expands to *device-to-host copy -> `cudaLaunchHostFunc` callback ->
host-to-device copy*. The callback runs the CPU MoE (a ggml `MUL_MAT_ID` graph over the DDR arena) and returns the already
weighted, reduced result. The CPU work therefore lives *inside* the captured CUDA graph, so the whole decode graph is a
single backend split (`splits=1`, `transitions=0` in the `[GGML-QWEN-FULL]` log) instead of ~50-100 scheduler splits.

**Important behaviours**

* Offload layers configured but this variable unset: full-graph decode is disabled with
  `[GGML-QWEN-FULL-DECODE] disabled: CPUOffload layers are configured, but the opt-in cudaHostNode bridge is disabled`
  and decode falls back to the layered path.
* A layer that also has a planned Option-B static hot/cold split keeps the split; the whole-layer marker is bypassed for it.
* Enabling it makes the CUDA schedule default to `blocking` (see below).
* A CUDA *conditional* node cannot contain host nodes (instantiate fails with `invalid argument`), so host-node mode cannot
  be combined with `ZEROMOE_QWEN4EXP_MASTER_GRAPH` conditional bodies. See howto section 10.
* Detailed Qwen graph and replay timing follows `--verbosity 5`. DEBUG timing can add synchronization overhead; the Phase 2 + host-node path has had profiling-only failures, so use it for diagnosis rather than performance measurement.

**Performance.** Not a throughput feature today. Plain scheduler-split decode: **28.0 tok/s**. All 26 CPU layers as host
nodes: **22.6 tok/s** (81%) with the blocking schedule, 18.2 tok/s with the old spinning schedule. Use it when you need a
single captured graph (the Phase 2 master-graph gate requires it), not for speed.

### `ZEROMOE_QWEN4EXP_HOST_NODE_PIN`

| | |
|---|---|
| Values | `0` disables; anything else / unset = enabled |
| Default | on, but only takes effect when `ZEROMOE_QWEN4EXP_HOST_NODE=1` |
| Effect | Pins the host-node CPU thread pool's N workers to the first N CPUs of the process's allowed set (ggml `strict_cpu`), and restores the main thread's affinity afterwards. Does nothing if fewer distinct CPUs than threads are available. |

Why: the callback runs on a CUDA driver thread that leads the OpenMP team, and unpinned workers land on arbitrary logical CPUs: the same graph computes ~33% slower
than from the main thread. Pinning gave host-node mode 22.70 -> 26.25 tok/s (+15.6%, identical output). It is deliberately not applied to plain scheduler-split modes
(0/44/4 got ~4% slower with binding, 22/26 ~4% faster, both within noise-to-negative). Do not combine host-node mode with `OMP_WAIT_POLICY=active`:
spinning workers plus a busy main thread can starve the callback thread (measured 7x slower in isolation).

### `ZEROMOE_QWEN4EXP_HOST_NODE_GPU_REDUCE`

| | |
|---|---|
| Values | `0` disables; anything else / unset = enabled |
| Default | on, only takes effect when `ZEROMOE_QWEN4EXP_HOST_NODE=1` |
| Effect | The callback returns the unweighted `[D, topk]` expert outputs (no CPU weighted-reduce tail, no weights D2H); the GPU does mul by router weights + sum over topk after the marker. |

Measured (6 interleaved runs each, idle box): 26.46 vs 26.03 tok/s (+1.65%, every ON run >= every OFF run except one). Output text identical in the checked pair; the reduction order differs from the CPU path so it is not guaranteed bit-identical.

## `ZEROMOE_QWEN_CPU_THREADS`

| | |
|---|---|
| Values | Positive integer N. Unset, empty, non-numeric or <= 0 = ignored. |
| Default | all OpenMP threads (`omp_get_max_threads()`: the `omp_max_threads` config key / `OMP_NUM_THREADS`; 12 on the reference box) |
| Effect | **Caps** (never raises) the threads used for Qwen4Exp CPU MoE work: the full-graph CPU backend, the host-node CPU runner, and the grouped CPU kernels. Effective value = `min(N, available)`. |

It is applied per call site and never calls `omp_set_num_threads()`, so attention and other model families keep the global
OpenMP setting.

Related: `ZEROMOE_QWEN_HOST_NODE_THREADS` caps only the host-node runner. It is consulted only when
`ZEROMOE_QWEN_CPU_THREADS` is unset.

**Guidance**

* Plain scheduler-split decode: leave unset (12). More threads were better in the earlier sweep (2 -> 6.9, 4 -> 8.3, 8 -> 14.0,
  12 -> 15.6-15.9 tok/s on the older configuration).
* Host-node mode with a *spinning* CUDA schedule: `11` recovers most of the loss (18.18 -> 22.10 tok/s, +21.6%) by leaving a
  logical CPU for the generation thread. With the blocking schedule (now the host-node default) it is not needed, and the two
  are not additive (22.03 combined).
* Do not combine with `GOMP_CPU_AFFINITY` pinning (measured slower).

## `ZEROMOE_CUDA_SCHEDULE`

| | |
|---|---|
| Values | `auto`, `blocking`, `spin`. Anything else, including unset or empty, selects the default. |
| Default | `auto` (`cudaDeviceScheduleAuto`). **Exception:** `blocking` when `ZEROMOE_QWEN4EXP_HOST_NODE=1` and this variable is unset/empty. An explicit value always wins. |
| Effect | Sets the CUDA device scheduling flags once per device at startup (`cudaSetDeviceFlags`). Logged as `[CUDA-SCHEDULE] device=N mode=...`. Controls how a thread waits in `cudaStreamSynchronize`/event waits: `auto` and `spin` busy-wait, `blocking` sleeps and is woken by an interrupt. |

**Why `auto` is the default.** With CUDA graphs on, blocking wake-up latency hurt full-graph decode (about 5.8 ms/token in
`cudaStreamSynchronize` vs 0.9 ms with `auto`), and llama.cpp itself never overrides the mode. In plain scheduler-split mode the
generation thread is part of the OpenMP team, so spinning costs nothing extra.

**Why host-node mode uses `blocking`.** The CPU callback runs on a CUDA driver thread that acts as OpenMP team master while the
generation thread waits in `cudaStreamSynchronize`. A spinning wait burns a 13th runnable thread on a 12-logical-CPU machine and
every OpenMP barrier stalls on stragglers. The wait happens once per token here, so the wake-up latency is irrelevant.
Measured: 18.18 -> 22.63 tok/s (+24.4%).

`spin` is available but has no measured use case.

The EXL3 executor also honors an explicit `ZEROMOE_CUDA_SCHEDULE` value.
With the variable unset, it preserves the current device flags. Its many
per-projection waits differ from the GGUF graph workload measured here;
use `exl3_hc_bench --omp-threads 12` to compare waiting policies with OpenMP
workers active before each HC dispatch.

For the standard EXL3 hybrid 26/22 profile, a warmed full-checkpoint check
measured about 6.40 tok/s with default OpenMP waiting and 7.10 tok/s with
`OMP_WAIT_POLICY=PASSIVE`, with all 128 generated token IDs preserved.
Use passive OpenMP waiting and CUDA `auto` for this measured EXL3 profile;
[the EXL3 comparison](exl3-wait-policy-2026-10-05.md) records the commands,
cache conditions and raw logs. The GGUF recommendations below are unchanged.

## Recommended combinations

| Mode | `HOST_NODE` | `QWEN_CPU_THREADS` | `CUDA_SCHEDULE` | `OMP_WAIT_POLICY` |
|---|---|---|---|---|
| Plain scheduler-split decode (22/26, Option-B 0/44/4) | unset | unset | unset (`auto`) | leave default (spinning) |
| Host-node / single-graph capture | `1` | unset | unset (auto-selects `blocking`) | leave default |
| A/B a host-node build without the fix | `1` | `11` | `auto` | leave default |

Leave `OMP_WAIT_POLICY` and `GOMP_SPINCOUNT` at their defaults: on 0/44/4, passive waiting cost 11% (30.43 vs 34.31 tok/s) and
`GOMP_SPINCOUNT=1000` cost 6%, even though a CPU-only microbenchmark favours passive (threads idle during GPU phases between CPU
splits, so waking them every split costs more than spinning saves).

Also check before benchmarking: another process using the CPU or GPU (a compile, `llama-bench`, a second `zeromoe`) inflates
or blocks results; the engine's preflight refuses to start if free RAM/VRAM is insufficient.

## Decode graph setup: arena borrowing and bootstrap (defaults)

By default the engine prepares reusable device arenas at startup and the first decode graph borrows them, so decode graph construction
takes ~0.1 s instead of 3+ s. This is controlled by two switches in the opt-out direction:

| Variable | Effect |
|---|---|
| `ZEROMOE_QWEN4EXP_GRAPH_BOOTSTRAP=0` | Skip the startup preparation (lazy: the first request pays the upload). |
| `ZEROMOE_QWEN4EXP_DECODE_RECREATE_GRAPH_ARENA=1` | Do not borrow arenas: every decode graph recreates and re-uploads its own (slow, ~3.2 s on Option-B splits). |

For Option-B per-expert split configs the startup step also uploads the ~13 GiB split arena once (+~2.7 s startup) and keeps it resident while prompts
are short; a long prompt (roughly more than ~120 tokens on a 24 GiB card) releases it before prefill and rebuilds it for that request's decode.
See `tests/option_b_split_reverify_howto.md` section 18 for measurements.
