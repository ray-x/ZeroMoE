# Ling reference audit inputs

See `docs/ling-3.1-flash-performance-audit-2026-10-20.md` for findings,
timing definitions, reference worktrees, and the failed diagnostic.

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
then uses `run_capped.sh` with `MemorySwapMax=0`. Its default is the audited
72 GiB DDR / 110 GiB cgroup baseline, GPU hot sets 3+12.5 GiB, CPU lane 10
threads, GPU assist 5+2, and pinned fraction 0.2. The monitor records memory
and GPU samples and stops only its own model if the memory floor is crossed.
It must run outside the restricted sandbox to observe the actual host.

An identical-token comparison with the trace-derived ranking:

```bash
LING_USAGE_FILE="$PWD/bench/ling/reference-audit/trace-first-reply.pin" \
LING_TOKEN_FILE="$PWD/bench/ling/reference-audit/baseline.tokens" \
LING_LOG=/tmp/ling-reference-forced.log \
python3 bench/ling/reference-audit/monitor.py
```

Omit `LING_TOKEN_FILE` for the required free-running check. Do not report a
forced-sequence improvement as an established free-running speed gain.

The selected-copy GPU-prefill diagnostic was verified with a smaller pool:

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
