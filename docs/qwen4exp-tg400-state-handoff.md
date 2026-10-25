# Qwen4Exp TG400 benchmark and state handoff

This note records the Q2_0 and IQ1_M decode-setup optimization measured on
2026-10-02. The reproducibility profiles are
`configs/qwen4exp_tg400_q2_0_22_26.toml` and
`configs/qwen4exp_tg400_iq1m_22_26.toml`.

## Machine and build

- GPU: NVIDIA RTX 3090, 24 GiB, SM 8.6.
- CPU: Intel Xeon W-2235, 6 cores / 12 threads.
- RAM: 128 GiB.
- Driver/CUDA toolkit: driver 610.57.04, CUDA 13.3.73.
- ZeroMoE build: Release, `-O3 -DNDEBUG`, native host CPU tuning,
  AVX-512 VNNI enabled, OpenMP enabled, CUDA architecture 86.
- GGML: the benchmark binary was built against an `upstream-master` llama.cpp
  checkout prepared on 2026-10-01. Its temporary `/tmp` checkout and build
  cache were removed; the exact upstream commit SHA was not preserved, so this
  record does not claim a pinned upstream revision. The tested Q2_0 CPU
  `quants.c` had the same SHA-256 in the Bee reference and ZeroMoE source:
  `9803e084b9861b3d72b2470423610129259d3843246be215e3e35ba5b7d222e6`.
- Bee reference: llama.cpp build `43fed6726` (build 11625), with the Q2_0 CPU
  VNNI patch enabled in its CPU backend.

The required graph-capture build settings are:

```text
CMAKE_BUILD_TYPE=Release
CUDA target=sm_86
ZEROMOE_USE_GGML=ON
ZEROMOE_USE_GGML_CUDA=ON
ZEROMOE_ENABLE_CUDA_GRAPHS=ON
GGML_CUDA_GRAPHS=ON
GGML_NATIVE=ON
GGML_OPENMP=ON
ZEROMOE_GGML_OPENMP=ON
ZEROMOE_ENABLE_AVX512_VNNI=ON
```

The CUDA compile invocation used the SM86 code target and the ggml graph
macros. The relevant nvcc flags were `-O3 -DNDEBUG -std=c++17
--generate-code=arch=compute_86,code=[compute_86,sm_86]
-DGGML_CUDA_USE_GRAPHS -DGGML_CUDA_KVARN -DGGML_SCHED_MAX_COPIES=4
-use_fast_math -extended-lambda -Xcompiler=-fPIC`. The upstream master
checkout's full `compile_commands.json` was in the deleted temporary build
tree, so this list records the flags captured during the run rather than a
reconstructed command for every translation unit.

`ZEROMOE_ENABLE_CUDA_GRAPHS=ON` is propagated to ggml as
`GGML_CUDA_GRAPHS=ON`, which compiles `GGML_CUDA_USE_GRAPHS`. The runtime log
must say `[CUDA-GRAPHS-CHECK] compiled in: yes`. Without that compile-time
macro, ggml launches graph nodes individually and this decode path is much
slower.

The source build command used for the measured binary was:

```bash
cmake --build /tmp/zeromoe-upstream-master-build-20261001 \
  --target zeromoe -j 8
```

That build directory is temporary. To build again, point
`ZEROMOE_GGML_ROOT` at the intended upstream ggml checkout and pass the
settings above when configuring. The missing GGML SHA is the main gap in
recreating the historical binary byte-for-byte.

## Placement and model configs

Both configs use the same scattered expert placement. CPU-affinity layers are
`[0,1,2,3,4,5,6,7,14,20,21,22,24,28,30,33,34,36,38,40,41,42,43,44,45,46]`.
The GPU candidate range is `[8,48)` with trailing placement enabled. The mask
leaves 22 complete MoE layers on the RTX 3090 and 26 on CPU/DDR.

Both profiles use context 416, 400 generated tokens, temperature 0, top-p
0.95, 12 CPU/OpenMP threads, all experts preloaded in DDR, 100 GB host RAM
budget, usage statistics frozen, `top_n=256`, and minimum coverage 0.80.
The configured GPU budget is 4.1 GB dense, 2.0 GB KV, and 11.5 GB MoE cache
with 4-bit KV. The model paths and complete TOML settings are in the two
profiles linked above.

## Fixed inputs and runtime switches

The runs used a 16-token prompt-ID file at
`/tmp/tg400_qwen_prompt_tokens.txt`, with FNV-64
`0x2f075b2eebfbf8a7`. Each quantization had a 400-token continuation captured
from a ZeroMoE run and then replayed by both engines:

```text
Q2_0:  /tmp/zeromoe_q2_0_tg400_decode_tokens_20261002.txt
IQ1_M: /tmp/zeromoe_iq1m_tg400_decode_tokens_20261002.txt
```

The fixed token files and raw `/tmp` logs were not committed. Preserve the
prompt-ID file and generate one matching continuation file per quantization
to repeat the test exactly. Runtime switches used for the measured runs:

```text
CUDA_VISIBLE_DEVICES=0
ZEROMOE_QWEN4EXP_FULL_GRAPH=1
ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE=1
ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=40
ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1
ZEROMOE_PROMPT_TOKEN_INPUT_FILE=/tmp/tg400_qwen_prompt_tokens.txt
ZEROMOE_TEACHER_FORCE_TOKEN_FILE=<quantization-specific 400-token file>
ZEROMOE_TEACHER_FORCE_SKIP_SAMPLING=1
ZEROMOE_BENCH_SILENT=1
ZEROMOE_BENCH_IGNORE_STOP=1
```

The prompt-token file overrides tokenization so both engines receive the same
16 IDs. Teacher forcing supplies the same 400 next-token IDs to each engine;
the target forward pass and logits are still computed. Sampling is skipped,
and silent mode removes token printing and per-token VRAM sampling from the
timed path. MTP was not enabled.

The ZeroMoE measured-run shape is:

```bash
{ printf 'TG400 timing run\n:q\n'; } | env \
  CUDA_VISIBLE_DEVICES=0 \
  ZEROMOE_QWEN4EXP_FULL_GRAPH=1 \
  ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE=1 \
  ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=40 \
  ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1 \
  ZEROMOE_PROMPT_TOKEN_INPUT_FILE=/tmp/tg400_qwen_prompt_tokens.txt \
  ZEROMOE_TEACHER_FORCE_TOKEN_FILE=/tmp/zeromoe_q2_0_tg400_decode_tokens_20261002.txt \
  ZEROMOE_TEACHER_FORCE_SKIP_SAMPLING=1 \
  ZEROMOE_BENCH_SILENT=1 \
  ZEROMOE_BENCH_IGNORE_STOP=1 \
  ./build-tg400/zeromoe --config configs/qwen4exp_tg400_q2_0_22_26.toml
```

Decode borrows the completed prefill graph's dense/expert arenas by default.
Set `ZEROMOE_QWEN4EXP_DECODE_RECREATE_GRAPH_ARENA=1` to explicitly use the
older behavior and create a separate decode arena.
Full-graph decode also prepares those prompt-independent arenas during startup
by default; set `ZEROMOE_QWEN4EXP_GRAPH_BOOTSTRAP=0` to restore lazy
preparation for a diagnostic run.

For IQ1_M, switch to its config and continuation file. To capture a fresh
continuation, set `ZEROMOE_GENERATED_TOKEN_FILE` and omit both teacher-force
variables for that capture run. Bee used
`BEE_LLAMA_BENCH_TOKEN_FILE` for the prompt IDs and
`BEE_LLAMA_BENCH_DECODE_TOKEN_FILE` for the matching continuation.

The Bee result row was `pp16+tg400`, three measured repetitions after a prompt
and generation warmup, 12 threads, `ngl=999`, and a tensor override that kept
the same 26 scattered MoE layers on CPU. The test context is 416: Bee's
llama-bench sets `n_ctx = n_prompt + n_gen + n_depth`.

The fixed decode continuation used a test-only addition to the Bee runner:
`BEE_LLAMA_BENCH_DECODE_TOKEN_FILE`. It replays the captured continuation
instead of choosing random next-token IDs. It is not a stock llama-bench
option. The benchmark invocation shape was:

```bash
BEE_LLAMA_BENCH_TOKEN_FILE=/tmp/tg400_qwen_prompt_tokens.txt \
BEE_LLAMA_BENCH_DECODE_TOKEN_FILE=/tmp/zeromoe_q2_0_tg400_decode_tokens_20261002.txt \
./llama-bench -m <q2_0-gguf> -p 16 -n 400 -r 3 -t 12 -ngl 999 \
  -lm mmap+mlock \
  -ot 'blk\.(0|1|2|3|4|5|6|7|14|20|21|22|24|28|30|33|34|36|38|40|41|42|43|44|45|46)\.ffn_(gate|up|down)_exps\.weight=CPU'
```

For IQ1_M, replace both model and decode-token paths with the IQ1_M files.

## What changed and why it helps

Before the change, decode setup freed the completed prefill scheduler and
seeded a new decode graph with many synchronous host-to-device state copies.
The copy path synchronizes the CUDA stream after each `ggml_backend_tensor_set`.
For this model that state seed alone took about 42.8 ms; full prefill-finish
to ready-decode-graph setup was about 126 ms.

The updated path builds an address table for the completed prefill state and
binds compatible recurrent tensors directly into the decode graph. It aliases
72 GDN history/state tensors plus one PLE history tensor (73 of 97 state-table
entries). The 24 dense-attention KV entries stay decode-owned because prefill
has only 16 rows while decode needs capacity 416. Keeping the prefill scheduler
alive also keeps those aliased buffers valid. This retains roughly 0.63 GiB of
prefill scheduler workspace in addition to the decode workspace, trading VRAM
headroom for lower first-token setup latency. The measured Q2_0 run had about
5 GiB free before the graph reset, and the IQ1_M run completed with the same
handoff without OOM.

## Results and timing boundaries

| Quant | ZeroMoE `[REQUEST-DECODE-WALL]` | Decode setup | State seed | Bee `pp16+tg400` row |
|---|---:|---:|---:|---:|
| Q2_0 | 26.58 tok/s | 53.9 ms | 0.3 ms | 27.70 ± 0.12 tok/s |
| IQ1_M | 26.68 tok/s | 56.6 ms | 0.3 ms | 27.24 ± 0.49 tok/s |

ZeroMoE's `[REQUEST-DECODE-WALL]` starts after `[PREFILL-DONE]`, includes
decode-graph setup and all 400 decode steps, and excludes prefill execution.
The first decode graph took 52.2 ms to build for Q2_0 and 44.9 ms for IQ1_M;
the remaining setup is arena/scheduler/PLE handling. The original Q2_0
measurement showed 126.0 ms total setup, including 42.8 ms state seeding.

Important boundary caveat: Bee's llama-bench stopwatch starts before
`test_prompt()` and stops after `test_gen()`. Its `t/s` numerator is
`n_prompt+n_gen=416`, so the Bee rows above include both prefill and decode.
The earlier comparison multiplied Bee's row by `400/416`, producing
26.63 tok/s for Q2_0 and 26.19 tok/s for IQ1_M. Those are token-count-scaled
aggregate rates, not isolated Bee decode rates. They yielded apparent gaps of
−0.21% and +1.86%, but they do not prove a strict decode-only <3% result.
ZeroMoE's raw decode-only rates are shown separately so the timer boundaries
stay explicit. A strict decode-only Bee comparison needs a Bee generation-phase
timer around `test_gen()` (including its first decode setup/capture cost),
reported separately from the existing combined `pp16+tg400` row.

Bee's warmup includes a prompt and one generation pass before the three timed
repetitions. Its TG400 row therefore also does not report cold
first-generation graph-capture cost. The result establishes the measured
steady-state rows and the ZeroMoE setup reduction; it does not contain a
Bee first-token/setup-latency measurement.

The cold graph-build time is a different measurement from request decode
setup. It includes expert lookup, file reads, and H2D copies whose byte volume
depends on the quantization and resident layer size. For example, the final
Q2_0 cold graph build copied 14.50 GiB of experts, while IQ1_M copied 17.74
GiB. The quantization-independent graph-transition comparison is the warm
decode setup and its recurrent-state seed above.

## Reproduction notes (2026-10-02, upstream integration ggml)

- The fixed token files and the teacher-force/silent harness were not preserved; the hooks
  (`ZEROMOE_BENCH_SILENT`, `_IGNORE_STOP`, `ZEROMOE_TEACHER_FORCE_*`) exist in
  `include/generation_loop.hpp`, but a plain free-running 400-token decode is enough.
- **The prompt must tokenize to exactly 16 tokens** (the chat template adds 12, so use a
  4-token prompt such as "Tell a story."). The profile's `context_size` is 416: with a 17-token
  prompt the last token overflows the KV capacity, drops to the legacy path (~1.4 s) and the run
  reads ~24 tok/s, a fake ~11% gap. Check `[PREFILL-DONE] 16 tokens`.
- Required switches beyond the doc's list: `ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=40`,
  `ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1`, `ZEROMOE_QWEN4EXP_FULL_GRAPH=1`,
  `ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE=1`, plus `..._BORROW_PREFILL_ARENAS=1` and
  `..._GRAPH_BOOTSTRAP=1` for the 54 ms setup (borrow alone fails the 1 GiB VRAM floor and falls
  back to a degenerate layered decode; neither flag works on per-expert split configs).
- Measured: ZeroMoE 26.17 / 26.44 / 26.89 tok/s over 400 tokens (setup included) vs Bee
  `llama-bench -p 0 -n 400 -r 3` 26.96-27.06 (build 43fed6726, same 26-layer CPU mask).
  Per token: graph run ~36.3 ms + ~0.5 ms sampling. GPU kernels ~12.9 ms (about 3,900 kernels),
  CPU MoE ~19 ms, ~5 ms across 53 CPU/GPU transitions.
- Tried and rejected here (no gain or slower): AVX-512 gather iq2_xxs kernel (3.6x slower),
  HC stream-mean fusion (0%), AVX-512 VNNI iq4_nl (memory-bound, ~8.3 GB/s per core).
