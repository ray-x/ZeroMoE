# Graph-native MTP: shared runtime, Qwen4Exp adapter

Status: the standalone draft graph is parity-validated against llama.cpp, and
target-verified speculative generation is available as an **opt-in**
(`mtp_graph = true`), correctness-validated path on the Qwen4Exp full graph.
It is exact (output identical to plain decoding) but **not yet faster**: the
target verifies one token per step. The startup rejection of `mtp_enabled`
with the full-graph-only trunk stays for every configuration that does not set
`mtp_graph`.

## Real-weight runtime smoke (2026-10-26)

The graph target built and replayed the Qwen3.8-Flash-Next UD-IQ1_M trunk
with its shared Q4_K_M MTP sidecar on an RTX 3090. With 32 synthetic
finite fixture rows it completed two graph builds and 96 replays across
reset and chunked-replay phases; weights used 1,892,929,280 device bytes.
The dump self-check passed its reset/chunk invariants. These inputs were
synthetic, so this is a runtime smoke, not numerical parity against
llama.cpp. The MTP reference fixture and exact reference checkout remain
required before enabling end-to-end speculative generation.

## Cross-engine parity finding (2026-10-26)

Against llama.cpp `6050b46` on the 32-row synthetic fixture, only logits at row
21 failed (cosine 0.99962, NRMSE 0.0275). Rows 19, 20 and 22 are bit-identical;
the NextN block has `compress_ratios[48] = 0`, so QSA is not involved.
The divergence was traced as follows:

- It follows the **data, not the position**: swapping the contents of rows 21
  and 25 makes the row-21 hidden state and logits match, and the error appears
  at an unrelated row (logits at row 23) with hidden NRMSE 1.5e-8.
- On a 96-row Gaussian fixture (OOD inputs), 39/96 rows exceed NRMSE 0.01 and
  some router IDs flip, yet argmax agrees on 93/96. Where the bridge is exact,
  `hc_mixed` (the attention input) is also bit-exact, while `attn_out` differs
  on about half of rows from position 24. The fault is therefore inside the
  attention op (or the activation quantization it feeds), not in the HC mix.
- Row 12 is separate: its bridge differs in one whole HC stream by ~1e-3,
  consistent with one flipped Q8 activation rounding upstream.
- ZeroMoE's dumps are bit-reproducible run to run.
- norm, quantize, flash-attention, mmvq, binbcast CUDA kernels are identical
  between ZeroMoE's ggml (8baabd5) and the reference; `rope.cu` differs only
  by the fused rope+set_rows cache write. The HC fused-vs-unfused hypothesis was
  tested: the CUDA backend fuses the unfused sequence the same way in both.

Root cause (confirmed 2026-10-26): the llama.cpp reference runs its
weightless `RMS_NORM` nodes (enorm, hnorm, attention Q-norm, K-norm) on the
**CPU backend**, while ZeroMoE runs them on CUDA. `GGML_SCHED_DEBUG=2` on the
reference shows the CPU splits (`norm-48` inputs) around each of them; the
weights and the following MUL/ROPE are on CUDA. The CPU norm accumulates in
double and the CUDA kernel in float with a different reduction order, so the
normalized values differ by about one float ulp (~6e-8 NRMSE) on most rows.
A flipped fp16 rounding of Q inside flash attention, or a flipped Q8
activation rounding in a later Q4_K matmul, then amplifies that to the
4e-4..8e-2 differences seen on a minority of rows, and the K/V cache carries
it into later rows.

Evidence:

- Raw Q/K projections (Q4_K) and V are bit-exact between engines; only the
  post-norm Q/K differ, so the matmul kernels are not the cause.
- The relevant CUDA kernels (`norm.cu`, `fattn-*`, `mmvq.cu`, quantize) are
  identical in both ggml revisions; the fused rms_norm+mul+rope kernel in the
  reference is not reached because the norm is not on CUDA.
- Forcing ZeroMoE's four norms onto the CPU backend (`ggml_backend_sched_
  set_tensor_backend`, experiment only) makes every stage bit-exact against the
  reference: all 32 rows of the smooth fixture (including row 21), all 48 rows
  of the Gaussian fixture (including row 12, whose bridge flip is the same
  effect in enorm/hnorm), covering attention, router weights, FFN, hidden and
  logits.

Conclusion: ZeroMoE's draft graph is numerically correct; the row-21 failure
is reference-placement noise. Production keeps the norms on CUDA. For the
strict parity gate, `qwen4exp_mtp::Options::reference_cpu_norms` (parity binary
flag `--reference-cpu-norms`) places the same four norms on the CPU backend,
as the reference does. With it the 32-row fixture passes the comparator with
every stage bit-exact (cosine 1, zero error, identical routes); without it the
original row-21 logits failure reproduces. Use the flag for the bit-exact gate;
a CUDA-placement run is expected to show sporadic NRMSE spikes and should be
judged by top-1 / router agreement with a bounded hidden error instead.

The scheduler's `reserve_size` estimate was about 0.5% low when nodes are forced
across backends (720,896 vs 717,568 device bytes). Admission now reconciles the
live allocation against the device budget and host guard instead of requiring
it not to exceed the estimate.

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

## Target verification driver (2026-10-26)

`models/graph/mtp_speculation.hpp` is the model-agnostic speculative loop. It
reaches the target and draft only through `SpecTarget` (step, save, restore,
release) and `SpecDraft` (step, rewind), so it is proven on the CPU against a
deterministic fake before any GPU run (`tests/test_mtp_speculation.cpp`).

Cycle: a pending token `y` the target has not processed; the draft proposes
`d1..dk` (pairing `y` with the preceding true target hidden, then its own
residual); the target steps `y, d1..dk` one token at a time with a checkpoint
before each step after the first; exact speculative sampling accepts `a`
drafts and draws the correction from `norm(max(0, p - q))` (or the bonus
token when all are accepted); the target is restored to the checkpoint taken
before step `a+1`, which is exactly the state after `d_a`, with no replay; the
draft rewinds to the pending position and rebuilds the accepted rows from the
true target hiddens.

Proven by the CPU test: greedy output is identical to plain decoding for
perfect, noisy and adversarial drafts and for `k = 0..5`; the first three
sampled tokens match the target's exact joint distribution (TV < 0.005 over
120k runs) for good and bad drafts; EOS, caller stop and capacity truncation;
target/draft failures propagate and every checkpoint is released. Six
deliberate bugs (skipped or wrong restore, residual not subtracted, draft
rebuilt with its own residual, always-accept, skipped rewind) each fail the
suite. `MtpGraphRuntime::rewind` is the draft's rollback; it is exact because
the draft owns only K/V.

This proves the loop, not the model integration: the Qwen target adapter
(wide-residual export from decode, device state checkpoints) and the GPU
teacher-forced validation remain, and the production startup guard stays.

## Graph-native generation (2026-10-27)

Enable with `mtp_enabled = true`, `mtp_graph = true`, `mtp_model_path`, and
`mtp_draft_tokens` in the TOML, with the full graph required as usual. The
pieces:

- `core/qwen4exp_graph_mtp.cpp` adapts the engine to the driver in
  `models/graph/mtp_speculation.hpp`: `GraphTarget` runs one token through the
  production decode path (embedding, HC expand, `layers_forward_gguf`) and
  reads back logits and the wide pre-head residual; `GraphDraft` wraps
  `MtpGraphRuntime`. Anything that is not the full graph is an error: there is
  no fallback to the layered speculator or to plain decoding once generation
  has started.
- `qwen4exp_wide_export` (set by `mtp_graph`) makes the decode graph read back
  `D*HC` floats per step and the prefill graph one row per prompt token.
- `qwen4exp_full_graph_state_save/restore/release` checkpoint the decode
  graph's GDN state, conv history and PLE history device-to-device, plus the
  host PLE window. KV is not copied: each step writes only row `position` and
  reads are masked to `<= position`. A rebuilt graph (for example KV capacity
  growth) invalidates checkpoints and fails the request with a message.
- Speculation uses the ordinary decode sampler (`Sampler::probabilities`,
  optional repetition penalty on generated tokens only), so verified drafts are
  accepted against exactly the distribution plain decoding samples. The older
  `MtpSpeculator::make_distribution` hard-codes a 1.3 penalty and is not used.

Admission order: the target decode graph is built first, then the draft is
admitted against the remaining device memory (about 2 GiB of NextN weights, K/V
and workspace). With 22 resident expert layers on a 24 GiB card the draft does
not fit and the request fails with a clear error; give the target fewer device
expert layers (`ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS`, 18 was used). Creating
the draft first starves the decode build, which is then declined, and the
target runs in a different configuration with different numerics.

Validation (Qwen3.8-Flash-Next UD-IQ1_M target, shared Q4_K_M MTP sidecar,
RTX 3090, greedy, 48 tokens, 18 resident expert layers):

- Output is byte-identical to plain decoding for `mtp_draft_tokens` = 1, 2, 3,
  4 and 5 (acceptance 62%, 49%, 36%, 31%, 30%; 11-18 state restores per run).
- Logits fingerprints: every position on the committed path is bit-identical to
  plain decoding for K = 1, 3, 5. `plain` is reproducible run to run.
- `tools/check_mtp_graph_equivalence.py` repeats this check on hardware.

Fixed on the way: `qwen4exp_full_graph_run` read prefill logits at column
`S-1`, which is out of bounds for Qwen4Exp (it builds logits for the last row
only) and aborted every full-graph Qwen3.8 prefill with a logits target. It now
reads the last column that exists.

Known limits: verification is one token per target step, so MTP currently costs
extra target steps for rejected drafts and gives no speedup; batched
verification (an S = k+1 continuation graph at position > 0) is the next step.
Teacher-forced per-frame comparison against llama.cpp for the target is not
covered here (the equivalence above is ZeroMoE-vs-ZeroMoE). Acceptance (30-62%)
was measured only with the UD-IQ1_M trunk and the Q4_K_M sidecar; another trunk
quantization will change it.

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
c++ -std=c++23 -O2 -Imodels/graph tests/test_mtp_speculation.cpp \
    -o /tmp/zeromoe-mtp-spec
/tmp/zeromoe-mtp-spec
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
    zeromoe.dump 0 4096 --reference-cpu-norms
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
