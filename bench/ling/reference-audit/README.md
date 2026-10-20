# Ling reference audit inputs

See [the benchmark method](../../../docs/ling-3.1-flash-benchmark-method.md)
for the current measurement contract, exact commands, timing definitions, and
recorded results. The 2026-10-20 handoff remains a historical working-state
note.

See `docs/ling-3.1-flash-performance-audit-2026-10-20.md` for the earlier
findings, reference worktrees, and the failed diagnostic.

- `prompt.txt`: exact raw prompt, `Hi there`.
- `baseline.pin`: original 10,396-entry usage file from the earlier runs.
- `trace-first-reply.pin`: 5,629-entry histogram of the baseline's actual
  full-graph decode routes. It fits that reply, not every free reply.
- `baseline.tokens`: its 32 generated token IDs, for controlled comparisons.
- `config.toml`: original 99 GB host budget, context 64, temperature zero,
  and two-card configuration. Graph residency comes from runner exports.
- `results-2026-10-20.json`: measured results and original log SHA-256 values.

Run from the repository root in a host shell with GPU/systemd access:

```bash
LING_LOG=/tmp/ling-reference-recheck.log \
python3 bench/ling/reference-audit/monitor.py
```

The runner checks other inference/build processes and physical memory,
then uses `run_capped.sh` with `MemorySwapMax=0`. Its default is a 64 GiB DDR
pool with a 95 GiB cgroup cap, GPU hot sets 3+12.5 GiB, CPU lane 10 threads,
GPU assist 3+1, pinned fraction 0.5, and 3700 pinned VRAM slots. The short
benchmark below passes `LING_DDR_GIB=72` explicitly; after master enabled
page-reusing eviction, 72 GiB long-prompt runs reached the 95 GiB cgroup cap,
so the safe runner default is 64 GiB. Its prefill pin profile is empty unless
`LING_PREFILL_USAGE_FILE` is set. The monitor records memory and GPU samples
and stops only its own model if the memory floor is crossed. It must run
outside the restricted sandbox to observe the actual host.

An identical-token comparison with the trace-derived ranking:

```bash
LING_USAGE_FILE="$PWD/bench/ling/reference-audit/trace-first-reply.pin" \
LING_TOKEN_FILE="$PWD/bench/ling/reference-audit/baseline.tokens" \
LING_LOG=/tmp/ling-reference-forced.log \
python3 bench/ling/reference-audit/monitor.py
```

Omit `LING_TOKEN_FILE` for the required free-running check. Do not report a
forced-sequence improvement as an established free-running speed gain.

The selected-copy GPU-prefill diagnostic was verified with a smaller pool.
The command below is historical: the current no-copy runner caps memory at
95 GiB and disables prefill op offload. Use the handoff command for current
performance work.

```bash
LING_DDR_GIB=64 LING_CAP_GIB=96 LING_STOP_GIB=26 LING_GPU_PREFILL=1 \
LING_USAGE_FILE="$PWD/bench/ling/reference-audit/trace-first-reply.pin" \
LING_TOKEN_FILE="$PWD/bench/ling/reference-audit/baseline.tokens" \
LING_LOG=/tmp/ling-reference-selected-copy.log \
python3 bench/ling/reference-audit/monitor.py
```

This requires the new C++ callback. It transfers selected expert planes
safely, but was slower on the two-token prefill and is not the default.
No CUDA kernel rebuild was needed. The callback regression test is
`tests/ggml_expert_copy_guard.cpp`; it uses protected unused source pages
to catch accidental whole-arena reads.

The regression was compiled with `g++ -std=c++17 -O2`, include paths
`backends/ggml` and `/home/ray/Projects/llama.cpp/ggml/include`, and linked
against `libggml-cpu` and `libggml-base` in `build-ling/ggml-build/src`.
It checks negative IDs, duplicate IDs, a consumer after the first graph
node, padding, and reset between graph executions. `results-2026-10-20.json`
also records the successful guarded copy run and the rejected pre-fix run.

For a longer prefill measurement, `run.sh` accepts `LING_PROMPT_FILE` and
`LING_CONTEXT_SIZE`. The runner logs the prompt path and SHA-256. To build a
prompt-matched usage profile, choose new output paths for `LING_USAGE_FILE`
and `LING_PREFILL_USAGE_FILE`, set `LING_FREEZE_USAGE_STATS=0`, and run once
with the prompt. Then run again with the same prompt and profile paths using
the default frozen statistics. This keeps the prompt used for profile
collection and measurement identical; the free-running decode check should
still omit `LING_TOKEN_FILE`.

`prompt-long-20261020.txt` and the preserved long route profile
`prefill-long64-ring-r05-20261020.pin` are the long-prompt inputs used for the
current 10 tok/s prefill result. `results-current-2026-10-21.json` records the
logs, hashes, and measured rates. `prefill-hi-there.pin` is a local generated
artifact and is not a stable short-prompt benchmark profile; check its entry
count before using it. Long prompts automatically enable the patched negative
route skip; see the method page for the crash this prevents.
