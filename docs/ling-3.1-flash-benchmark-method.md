# Ling 3.1 Flash benchmark method and recorded results

This page records **how** the Ling 10 tok/s decode / 20 tok/s prefill goal is
measured, where each number comes from, and the results that were actually
observed. The 2026-10-20 handoff is kept as a historical working-state note;
this page is the current measurement contract.

The goal is **not met**. The best recorded long-prompt prefill is 10.00 tok/s
for the prefill pass alone (8.85 tok/s including graph construction), and the
best short-prompt decode is 10.03 tok/s over the output interval (9.96 tok/s
including the one-time decode-graph setup). Do not report either number as a
completed 20/10 result.

## What counts as the target

The target is user-visible throughput, not a graph-only kernel rate:

- **Decode:** `REQUEST-DECODE-OUTPUT` for a free-running 32-token reply.
  `(tokens - 1) / (last_output_time - first_output_time)`. This excludes the
  prompt prefill and the first prefill output, but includes the one-time decode
  graph setup because setup happens between the first and second output.
  Report `REQUEST-DECODE-WALL` beside it: that timer starts at `PREFILL-DONE`
  and explicitly includes decode graph setup.
- **Prefill:** `PREFILL-DONE` for the exact prompt. `prompt_tokens / wall`.
  This is the prefill pass only, not graph construction.
- **Prefill including setup:** `prompt_tokens / (graph_prepare_seconds +
  prefill_seconds)`, where `graph_prepare_seconds` comes from
  `graph prepared for S=<tokens> in <ms>`. The stated objective includes graph
  setup, so this is the conservative number to use against 20 tok/s.
- **Do not use** the shutdown `Rates: decode=...` or `[REQUEST]` line as the
  target metric. The shutdown `decode` rate includes a different interval and
  can include prefill/transition time; `[REQUEST]` mixes prefill and decode.

The graph-only decode rate (`1000 / mean_step_ms`) is useful for finding a
kernel or scheduling regression, but it is not the user-visible rate.

## Workloads

Both workloads use concurrency 1, temperature 0, greedy output, and 32 new
tokens. The prompt file and its SHA-256 are logged in the `[RUN]` line.

| Name | Prompt | SHA-256 | Context | Prompt tokens | Usage profile |
| --- | --- | --- | ---: | ---: | --- |
| short | `bench/ling/reference-audit/prompt.txt` (`Hi there`) | `ba9d5a6258cef659c36fc0193a6c8c0d33a47762ff99798b01662e306f3ab942` | 64 | 2 | `usage-hi-there-robust4.pin` |
| long | `bench/ling/reference-audit/prompt-long-20261020.txt` | `cabf427daba5ff2f072a5ebb91fa1a2e437f858fddabda609649576ad90daa07` | 256 | 177 | optional prompt-matched `prefill-*.pin` |

`prefill-hi-there.pin` is a **local generated file**, not a fixed benchmark
input. It was overwritten during long-prefill profiling and must not be used
for short decode comparisons without checking its entry count. The preserved
long route profile is `prefill-long64-ring-r05-20261020.pin`. The runner leaves
the prefill profile empty unless `LING_PREFILL_USAGE_FILE` is set.

## Exact short-decode command

Run from the repository root on the host, with no other MoEStream, llama.cpp,
build, or sweep process running:

```bash
LING_DDR_GIB=72 LING_CAP_GIB=95 LING_STOP_GIB=20 \
LING_GPU0_CACHE_GIB=3.0 LING_GPU1_CACHE_GIB=12.5 \
LING_PINNED_VRAM_CAPACITY=3700 LING_OFFLOAD_PINNED=0.5 \
LING_GPU_ASSIST=3 LING_ASSIST2_SLOTS=1 LING_CPU_THREADS=10 \
LING_USAGE_FILE="$PWD/bench/ling/reference-audit/usage-hi-there-robust4.pin" \
LING_LOG=/tmp/ling-short.log \
bench/ling/reference-audit/run.sh
```

The runner writes the effective TOML to `/tmp/ling-reference-config.*.toml`,
logs the prompt SHA-256, creates `/tmp/ling-short.tokens` and
`/tmp/ling-short.routes`, and runs under `run_capped.sh` with
`MemoryMax=95G` and `MemorySwapMax=0`. The runner preflight rejects a run if a
model/build process is present or host `MemAvailable` is below 120.5e6 KiB.

For a prompt-matched long run:

```bash
LING_PROMPT_FILE="$PWD/bench/ling/reference-audit/prompt-long-20261020.txt" \
LING_CONTEXT_SIZE=256 \
LING_PREFILL_USAGE_FILE="$PWD/bench/ling/reference-audit/prefill-long64-ring-r05-20261020.pin" \
LING_DDR_GIB=72 LING_CAP_GIB=95 LING_STOP_GIB=20 \
LING_GPU0_CACHE_GIB=3.0 LING_GPU1_CACHE_GIB=12.5 \
LING_PINNED_VRAM_CAPACITY=3700 LING_OFFLOAD_PINNED=0.5 \
LING_GPU_ASSIST=3 LING_ASSIST2_SLOTS=1 LING_CPU_THREADS=10 \
LING_USAGE_FILE="$PWD/bench/ling/reference-audit/usage-hi-there-robust4.pin" \
LING_LOG=/tmp/ling-long.log \
bench/ling/reference-audit/run.sh
```

Long prompts (`LING_CONTEXT_SIZE > 64`) automatically set
`MOESTREAM_GLM5NEXT_BATCHED_NEGATIVE_SKIP=1`. Without the patched negative-id
skip, the valid-dummy route fallback aliases many prompt tokens onto expert 0
and the batched id compaction over-runs that expert's row range; the observed
failure was `CUDA error: an illegal memory access was encountered` in
`launch_mul_mat_q` at layer 4. Short-prompt runs keep the historical behavior
(`0`) so a two-token result is not silently changed.

## How the run is recorded

For every reported number, preserve all of:

1. The full `LING_LOG` (stdout plus timestamped engine lines).
2. The effective config path printed in `[RUN]`.
3. The token and route sidecars `*.tokens` and `*.routes`, with SHA-256.
4. The exact prompt path and SHA-256 from `[RUN]`.
5. The usage profile path and SHA-256. A profile is collected once with
   `LING_FREEZE_USAGE_STATS=0` on the same prompt, then reused frozen for the
   measured run.
6. The run's `[MEM-GUARD]`, `[CACHE-CHURN]`, GPU free-memory, and cgroup
   samples. Use `monitor.py` when a minimum-available-memory and swap record
   is required; `run.sh` alone does not produce a complete memory trace.

The checked-in `results-handoff-2026-10-20.json` and
`results-current-2026-10-21.json` are manifests, not a replacement for the raw
logs. Raw logs stay under `/tmp` because they contain host-specific paths.

Timing lines used here:

```text
[GGML-GLM5NEXT-FULL] graph prepared for S=<n> in <ms>
[PREFILL-DONE] <n> tokens in <s> (<rate> tok/s)
[REQUEST-DECODE-WALL] tokens=32 wall=<s> rate=<rate> tok/s (... includes one-time decode-graph setup)
[REQUEST-DECODE-OUTPUT] tokens=32 intervals=31 wall=<s> rate=<rate> tok/s (... first output to last output ...)
[GLM5NEXT-LANE] tokens=16 ... split=... misses=... loads=... read=... ms/token
```

Do not compare runs with different prompts, profiles, hot-cache bytes, pin
counts, or cgroup caps. A decode improvement is only established if the
free-running token/route sidecars are compared and the free-running result is
reported, not only a forced-token run.

## Recorded results

All rows use the 95 GiB cap, zero cgroup swap, and the 20 GiB stop floor. The
short rows use the exact 3700-pin configuration above. The long rows are
different cache/ring configurations and are not an A/B of prompt content
alone.

| Run | Graph prepared | Prefill pass | Prefill incl. setup | Decode output | Decode wall | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| short, historical dense reuse | 2113 ms | 1.024 s (2 tokens) | 0.63 tok/s | **10.03 tok/s** | **9.96 tok/s** | 31 intervals, 3.092 s; `usage-hi-there-robust4.pin`; 3700 pins |
| short, clean empty prefill profile | 2203 ms | 1.078 s (2 tokens) | 0.61 tok/s | **9.84 tok/s** | **9.78 tok/s** | current 3700-pin layout; empty prefill profile |
| long, ring r05, no borrow | 2291 ms | **17.695 s (10.00 tok/s)** | **8.85 tok/s** | 3.33 tok/s | 3.40 tok/s | 177 prompt tokens; ring 2.888 GiB GPU0 + 1.709 GiB GPU1 |
| long, cache borrow 3.0 GiB | 3262 ms | 19.591 s (9.03 tok/s) | 7.75 tok/s | 4.54 tok/s | 4.61 tok/s | 3.0 GiB GPU0 cache detached to host; 3700 pins; restore before decode |

The short history reached 10.03 tok/s output / 9.96 tok/s wall, but the latest
clean run with the current pin layout is 9.84 / 9.78. That 0.19 tok/s gap is
below the stated target and must not be rounded up to 10. The long-prefill
target is far away: 10.00 tok/s prefill pass / 8.85 tok/s including setup
against 20 tok/s.

The long ring run copies the cold experts of every layer into two reusable
ring sets, not only the experts selected by the prompt. At 177 tokens that is
roughly 111 GB of PCIe traffic and is the main reason prefill is near 10 tok/s.
Filtering the copies to prompt-used experts, while keeping the on-demand
fallback, is the next proposed optimization; it has not been measured yet.

## Reproduction constraints

- Do not merge this branch on the strength of a graph-only rate, a forced-token
  reply, or a run overlapping another model process. The current objective
  check fails on prefill.
- Do not reset, clean, or discard `build-ling/` or generated pin profiles; they
  are local benchmark artifacts.
- Keep the host stop at 20 GiB and `MemorySwapMax=0`. If `MemAvailable` drops
  below the preflight floor, wait for the other process to exit; do not lower
  the floor to make a run start.
- Rebuild C++ with `cmake --build build-ling -j 10` after source changes. The
  committed source must build before any measurement is claimed.

## Current status

- Best short decode: **10.03 tok/s output / 9.96 tok/s wall** (historical);
  latest clean **9.84 / 9.78**.
- Best long prefill: **10.00 tok/s pass / 8.85 tok/s including setup**.
- Target: **10 tok/s decode / 20 tok/s prefill including setup**.
- Status: objective not met; no merge has been performed. The branch is
  committed and rebased for review, but the 20 tok/s prefill gate remains open.
