# Graph-native MTP: shared runtime, Qwen4Exp adapter

Status: experimental standalone draft path. This is not end-to-end speculative
generation support. Keep the full-graph MTP startup rejection until target
verification and state reconciliation are implemented and validated.

## Reference

The reference is llama.cpp commit
`6050b467c07a95c7b87debd4ddac2f8ba8d9b807`, inspected in the checkout named
by `LLAMA_ROOT` when building the reference harness:

- `src/models/qwen4exp.cpp`, target `graph`: exports `t_h_nextn` before
  the final HC mixer, independently of logits selection.
- `src/models/qwen4exp.cpp`, `graph_mtp`: embedding/hidden bridge, ordinary
  HC-wrapped attention and FFN, wide residual output, then NextN head.
- `common/speculative.cpp`, `common_speculative_impl_draft_mtp`:
  model-independent `process`, `draft`, and `accept` orchestration.
- `src/llama-context.cpp`: unmasked target NextN embeddings retain every
  token row, even when only the last token requests logits.

These are two execution graphs, not two independent implementations of the
model equations. The target graph verifies proposals; the small draft graph
executes the NextN block. Both reuse applicable building blocks and weights.
GGML's CUDA backend owns capture; do not externally capture scheduler calls.

## Scope and boundaries

The first slice is a single-sequence, single-head, text-only draft session.
Fixed-shape, one-token replay serves both prompt catch-up and drafting.
Batched catch-up is an optimization, not a prerequisite. GLM and DeepSeek
adapters, multiple heads, shared-target KV, and multi-sequence scheduling are
not designed here.

Shared code under `models/graph/` owns request validation, session lifecycle,
input/output transfer, scheduler replay, memory admission, and errors.
`GraphCore` remains the allocation/handle foundation. Existing routed-MoE
operations remain the only expert graph implementation.

Qwen code owns the checkpoint contract, HC equations, attention parameters,
and graph assembly. Target and draft must reuse the same HC and attention
operations. Do not construct an unused target `FullGraph` just to call helpers.
Do not add a second legacy expert executor.

`FullGraphModel` continues to describe the target whole-model graph. Draft
requests have an explicit hidden-state input and their own cache; they are
not ordinary target prefill/decode requests.

## Input/output and alignment

Let `D` be embedding width, `W` the draft hidden width, and `V` vocabulary size.
For Qwen4Exp, `W = D * HC`; never assume `W == D`.

A replay consumes an embedding `[D]`, hidden input `[W]`, and position. It
returns logits `[V]` and hidden output `[W]`. Token lookup is outside the
graph, using the checked resident embedding; this matches llama.cpp's
embedding-input path. Norms, bridge, attention/KV update, HC, router, FFN,
and output projection are graph operations. Sampling is outside the runner.

Catch-up pairs token `x[p]` with target hidden `H[p-1]`; the first row uses
zero pending hidden. Carry `H[last]` across caller chunk boundaries.
The first actual draft invocation consumes an **already sampled target
token**, paired with its preceding target hidden, and predicts the following
token. Subsequent invocations feed back the previous draft residual.
Do not replace the target seed by sampling MTP logits for the same position.

The graph assembles:

1. RMS-normalize the embedding and each incoming HC stream, applying their
   respective learned gains.
2. Repeat the embedding across streams, concatenate `[embedding; hidden]`
   on the `D` axis, and project `2D -> D` per stream.
3. HC attention mix, Qwen attention, HC combine.
4. HC FFN mix, shared routed-MoE operations and gated shared expert, HC combine.
5. Export the wide residual **before** the NextN head mixer.
6. NextN HC head and the shared vocabulary projection.

## State and lifetime

The draft owns attention K/V with fixed addresses and a logical valid prefix.
Physical capacity is padded to 256 rows for CUDA flash-attention kernels;
padded/future rows are masked and do not extend the admitted logical context.
It never advances target GDN, convolution, PLE, HC, or target KV state.
Replay appends at `next_position`; reject gaps, rewinds, and capacity overflow.
Reset synchronizes work and clears the draft prefix. Unsupported rollback
must not be approximated by changing only a position counter.

Validate shapes and finite inputs before submitting. Publish output and
advance the logical position only on success. A failed submission or invalid
output can follow a partial KV update: poison the session until reset rather
than pretending rollback succeeded. Invalid requests before submission do
not poison an otherwise valid session.

Output storage is caller-owned; no tensor pointer escapes the runner.
Schedulers are destroyed before their arenas and borrowed owners. Keep
engine-resident bindings alive for the session. Rebuild on address,
placement, device, or model changes; never retarget captured pointers.

## Checkpoint, placement, and admission

Qwen's extra block lives in the separate MTP file at `blk.<trunk_layers>`,
not in the trunk GGUF tensor map. Validate its layer count, architecture,
dimensions, norm gains, expert geometry, and attention/rope metadata.
Existing resident loading is reused; packed MTP expert planes must not
enter trunk-file expert streaming.

Use shared embedding and vocabulary-head storage. Optional checkpoint
overrides must be explicitly supported and validated or rejected. No second
vocabulary-sized device upload is implicit.

An admitted graph reports incremental weights, KV and scheduler workspace
separately from borrowed storage. Admission occurs with the target graph
resident. A failed allocation is an explicit failure, not permission for
layered execution. Captured bindings are immutable.

### Interaction with pinned experts and the LRU

Draft experts are a separate, fixed resident allocation, not trunk expert
IDs in the RAM/VRAM LRU. Draft replay does not change usage counters, pin
rankings, `pin_topk_per_layer`, or eviction order. Trunk pinning policy remains
unchanged. The initial graph uploads the NextN block's packed experts; it does
not pin every expert of the target model.

The draft still competes for physical memory. Its host source weights count
against `MemoryGuard`; its incremental device weights, private KV, and
measured workspace require an explicit device budget. Future production
startup must admit these alongside target dense weights, pinned experts,
active-layer storage, target workspace, and an LRU/safety reserve. Creating a
draft after the target cache has consumed that space must fail admission,
not silently unpin target experts. This standalone slice does not resize the
target cache.

Verification batches may reuse experts across candidate tokens, but rejected
proposals also consume target compute and cache traffic. Neither reduced
cache pressure nor a decode speedup follows from MTP alone; measure acceptance,
expert hit/miss rates and end-to-end accepted tokens/second together.

## Attention compatibility

The inspected llama.cpp MTP graph calls the ordinary Qwen attention builder,
including IMRoPE and optional QSA/indexer pooling. Existing ZeroMoE MTP uses
dense attention. A dense implementation is valid only while the reference
selection retains all causal positions. Derive that bound from the actual
checkpoint metadata and reject larger contexts. A layer with no compression
does not need indexer selection.

Use the family attention implementation and verify its exact positional
convention against the reference. Do not import another family's RoPE
assumptions. Unsupported sparse-attention or positional configurations must
fail before state changes.

## Validation and rollout

The standalone harness supplies identical embeddings, hidden inputs and
positions to both implementations. llama.cpp's `graph_mtp` is authoritative;
the manual `qwen4exp_mtp_forward` is only a secondary diagnostic.

Compare bridge output, post-attention residual, router IDs/weights,
post-FFN residual, wide output, and logits at **every** replay. Fixed-token,
teacher-forced comparisons avoid divergent routes from free-running text.
For matched quantized CUDA placements require cosine >= 0.999 and normalized
RMSE <= 0.01 per row; report maximum absolute error and routing differences.
These are proposed acceptance gates, not measured results. Synthetic
representable kernel tests additionally require error <= 1e-4 against float64.

Cover 32 or more replays, chunked versus unchunked catch-up, reset/repeat,
invalid shape/token/position, context limits, missing tensors, denied memory,
poison/reset, and shutdown. Check fixed-shape graph reuse, unchanged target
state, and no legacy expert dispatch. Exported reference rows must identify
the exact checkpoint, reference revision, input bytes and placement.

Local CPU contract and fixture tests do not establish CUDA parity. Full
compilation, graph capture/replay and reference numerics require Linux with
NVIDIA CUDA and the matching GGML build. Check competing inference processes
and available GPU/host memory before model runs.

Only a later slice may remove the production guard. That slice must implement
target-graph multi-token verification, rejected-prefix state reconciliation,
chunked target hidden export, generation error propagation, and teacher-forced
per-frame validation, with no layered target decode.

## Standalone implementation and reproduction

The experimental entry point is `qwen4exp_mtp::create` in
`models/qwen/qwen4exp_mtp_graph.hpp`. The caller supplies shared ownership of an
engine containing the trunk metadata/head and resident NextN tensors loaded
from the separate MTP file. The runner retains that owner. It copies NextN
weights into private GPU storage, borrows the CPU vocabulary head without
uploading a second copy, and owns independent K/V. This intentionally
conservative placement is not a throughput-tuned production policy.

Creation builds and destroys a throwaway measurement graph/scheduler, then
constructs the live graph from scratch. Two construction calls at startup
are expected; replay and reset must not increase that count. Measurement
and live allocations are never simultaneously resident.

The fixture is ASCII. Its header is
`ZEROMOE_MTP_FIXTURE_V1 D W V N`, followed by N rows, each containing
`token_id position`, D embedding floats and W incoming hidden floats.
Positions must be `0..N-1`; N must be at least 32 for the parity gate.
All values must be finite. Supply actual target hidden rows with the
alignment above; do not feed each implementation its own output as input.
The token ID identifies the supplied embedding; these harnesses use
embedding-input execution and do not test embedding lookup.

Each exporter runs baseline replay, reset with caller chunks of seven rows,
and reset/repeat. Since the graph is S=1, chunking changes caller grouping,
not GGML batch shape. This tests persistence across chunk boundaries, not
future target-to-draft hidden-carry integration. The ZeroMoE exporter also
checks host target-state sentinels and unchanged construction count; it does
not establish coexistence with a running target CUDA graph.

Build and run the dependency-free session and comparator tests:

```sh
c++ -std=c++23 -Imodels/graph tests/test_mtp_graph_contract.cpp \
    -o /tmp/zeromoe-mtp-contract
/tmp/zeromoe-mtp-contract
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_mtp_parity.py
```

The standalone llama.cpp exporter builds without modifying its checkout:

```sh
cmake -S tests/mtp/llama -B build-mtp-reference \
    -DLLAMA_ROOT=/path/to/llama.cpp -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON
cmake --build build-mtp-reference --target llama_mtp_reference qwen_mtp_ops_test -j
./build-mtp-reference/qwen_mtp_ops_test
```

For the portable operation tests on macOS, use `GGML_CUDA=OFF`,
`GGML_METAL=OFF` and `GGML_OPENMP=OFF`. The operation test compares HC against
float64 and exercises persistent attention KV/masks over 32 replays and reset
with exactly representable values; it is not a real-model RoPE/parity test.

The reference exporter requires a **self-contained reference GGUF exposing
both the trunk embedding/head and the identical NextN block**, as accepted by
the pinned llama.cpp model loader. It does not merge a shared-file checkpoint.
Preparing that reference artifact, recording checkpoint hashes, and obtaining
real target hidden fixtures remain prerequisites for real-model parity.
The ZeroMoE exporter accepts the original separate trunk and MTP files.

After checking competing inference processes and available memory, on Linux
with CUDA:

```sh
cmake -S . -B build-mtp -DCMAKE_BUILD_TYPE=Release \
    -DZEROMOE_USE_GGML=ON -DZEROMOE_USE_GGML_CUDA=ON
cmake --build build-mtp --target qwen4exp_mtp_parity -j
./build-mtp/qwen4exp_mtp_parity trunk.gguf mtp.gguf fixture.txt \
    zeromoe.dump 0 4096
./build-mtp-reference/llama_mtp_reference reference-with-mtp.gguf \
    fixture.txt llama.dump 48 cuda
python3 tools/compare_mtp_graph.py llama.dump zeromoe.dump
```

`4096` is an explicit example draft budget in MiB, **not** a measured
requirement; use the admitted budget for the actual device/model. `48` is the
example trunk-layer count, validated against the supplied reference model.
Reference placement overrides only the NextN block to CUDA and keeps the
vocabulary head on CPU. Dumps include the input fixture fingerprint and
dimensions; provenance must additionally record model hashes, revision,
backend build flags, device and command lines.

The comparator requires every diagnostic row, exact routed IDs, per-row
cosine/NRMSE gates and bitwise reset/chunk repeatability. A missing or
incomplete dump is a failure. These programs and CPU tests do not constitute
evidence that the CUDA engine builds or that checkpoint parity has passed.
Production speculative generation remains rejected.
