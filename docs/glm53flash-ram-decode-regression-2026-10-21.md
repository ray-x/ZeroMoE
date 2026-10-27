# GLM 5.3 Flash RAM decode: early-ready selection

The whole-layer RAM profile regressed when the early-ready CPU lane became an
unconditional default. The lane's original adoption result was measured on an
offload workload; it does not establish a benefit for CPU-only expert branches
whose weights are already resident in RAM.

## Controlled comparison

All four runs used the same pre-fix R2 binary, ggml libraries, prompt, six CPU
threads and `glm53flash_AJ_IQ2XXS_wholelayer8_cpu.toml`. Layers 3-10 were in VRAM,
and layers 11-44 were preloaded into RAM. **SSD offload was disabled.** Loading
the model at startup is separate from decode-time demand loading; the lane runs
reported zero demand misses, loads and read time.

Order was ordinary graph / early-ready lane / early-ready lane / ordinary graph.
Every run generated the same 150 tokens with greedy sampling. Timing used
`--verbosity 5` (the recorded runs used the now-retired `ZEROMOE_TOKEN_LOOP_DIAG=1`); steady medians below exclude steps 0-19.

| Run | Early-ready | Request output tok/s | Steady median ms/token | Steady median tok/s |
|---|---|---:|---:|---:|
| normal_a | forced off | 6.66 | 137.147 | 7.29 |
| lane_b | forced on | 6.28 | 154.171 | 6.49 |
| lane_c | forced on | 6.31 | 153.655 | 6.51 |
| normal_d | forced off | 6.78 | 137.668 | 7.26 |

The ordinary graph improves the paired steady median rate by about 12%, and
the short-request output rate by about 6.8%. These are different metrics:
`REQUEST-DECODE-OUTPUT` includes the first interval's graph preparation and all
timing outliers; the historical 7.26 tok/s reference is a steady per-step median
(`GLM5.3-Flash-baseline.md`, section 7). The slower result is also present on
master, so comparing R2 only with current master cannot identify this older
default-selection regression.

## Selection fix

With `ZEROMOE_GLM5NEXT_EARLY_READY` unset, each layer uses the lane when it has
offload work or a GPU hot-expert branch to overlap. A fully RAM-resident CPU-only
branch uses the ordinary graph. Explicit `0` still disables the lane and `1`
still forces it, retaining an independent way to exercise the strategy lifecycle.

This selection uses placement and residency, rather than a model-name special
case. It does not change the selected experts, router weights or VRAM budget.
Per-expert hot/cold splitting across layers remains supported; the whole-layer
profile was held fixed here to isolate execution strategy from placement.

The frozen performance band was not widened.

## Fixed-default validation

After rebuilding, a 300-token run with `ZEROMOE_GLM5NEXT_EARLY_READY` **unset**
selected the ordinary graph (6131 nodes, zero early-ready lanes). It completed
successfully with a 139.076 ms/token steady median over steps 50-298, or
**7.19 tok/s**, within 1% of the historical 7.26 tok/s reference. Whole-request
output throughput was **6.83 tok/s**; this includes setup and timing outliers
and must not be substituted for the steady-median measurement.

The first 150 generated tokens matched all four pre-fix comparison runs. There
were no decode-time disk-fetch misses and no SSD-offload store was configured.
Both `zeromoe` and `zeromoe_tests` rebuilt successfully. The regression gate
now reports the steady median; it measured **7.21 tok/s**, above the 6.98 floor.
The native suite passed 168 tests with the same five pre-existing failures
(Tests 12/14/15/17/22), none new. A selector check confirmed the automatic
default and the legacy explicit `0`, `1` and empty-value override behavior.

## Reproduction

Use the profile above and the prompt:

```text
Explain the difference between CPU and GPU architecture in terms of parallelism, memory hierarchy and scheduling, in detail.
```

Run on an otherwise idle machine with `CUDA_VISIBLE_DEVICES=0`,
`ZEROMOE_GLM5NEXT_FULL_GRAPH=1`, `ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=512`,
`ZEROMOE_GLM5NEXT_CPU_THREADS=6`, `--verbosity 5` (then `ZEROMOE_TOKEN_LOOP_DIAG=1`), and
`ZEROMOE_BENCH_IGNORE_STOP=1`. Set `ZEROMOE_GLM5NEXT_EARLY_READY=0` or `1`
for the controlled comparison; unset it to validate the fixed default. Use
`scripts/run_capped.sh --limit-gb 100 -- <binary> --config <profile> --temp 0
--ngen 150 --force-start --log-path <unique-log>`.

The table above is the durable record; raw logs, token files and binary
manifests from the investigation were kept only on the measurement host.
