# DeepSeek V4.1 Flash graph

Checkpoint: 2026-10-15. Worktree: `/home/ray/work/moestream-deepseek-v41-flash`,
branch `deepseek-v41-flash`, base `1842a5c`.

## Verified behavior

Full graph prefill and decode matched **all 192 full logit-vector hashes
bit for bit** against fully native prefill and decode. The comparison used
three different prompts in one process, 64 generated tokens per prompt,
context 4096, temperature zero, and live caches across requests. The prompts
covered MoE explanation, arithmetic, and Python factorial code. A separate
comparison with graph prefill and native decode also matched all 192 hashes.
This establishes equivalence to the native engine on these cases; it is not
an independent model-quality comparison against upstream inference.

The tuned dual-GPU graph measured:

| Prompt | Graph decode | Fully native decode |
| --- | ---: | ---: |
| MoE explanation | 5.93 tok/s | 3.77 tok/s |
| Arithmetic | 7.10 tok/s | 4.20 tok/s |
| Python factorial | 4.62 tok/s | 3.23 tok/s |
| Combined, 192 tokens | **5.71 tok/s** | **3.69 tok/s** |

These use `[REQUEST-DECODE-WALL]`, from prefill completion through the last
output, including first-use decode setup. Prefill and startup preload time
are excluded. The code prompt remains below 5 tok/s; the target is reached
for the combined workload, not guaranteed for every prompt. Prefill speed
has not been optimized in this decode-focused work.

Use `configs/bench/deepseek_v41_100gib_ddr_17gib_vram_dual_gpu.toml`. It retains the
3090's 17 GiB main expert pool and approximately 100 GiB DDR capacity, adds
the 5060 Ti's 13.5 GiB main expert pool, and splits the existing 64 active
slots into 48 on the 3090 and 16 on the 5060 Ti. Decode async prewarm is off:
it previously issued transfers with insufficient lead time and reduced
throughput. Usage statistics stay frozen during comparisons.

## Correctness and performance fixes

- **Stable routed identity:** RAM-to-VRAM MOVE promotion clears a RAM slot's
  mutable expert ID. Dispatch and compute holds now use the token's routed
  ID. This fixed a measured 22% layer-0 MoE output mismatch.
- **Native HC semantics:** graph pre/post hooks and final collapse preserve
  V4.1's carried pre-mix. GLM's immediate collapse is not interchangeable.
- **Shared dense ownership:** graph mode previously skipped native dense
  upload, although its callbacks still called `dense_proj`. Shared experts,
  routers, and attention projections consequently fell back to CPU dequant
  despite the routed-expert counter reporting 100% GPU. V4.1 now uploads one
  native dense pool and lets graph tensors borrow its device pointers.
  CPU dequant/dot counters became zero, and decode rose from 3.16 to about
  4.8 tok/s before the active-pool tuning.
- **Exact grouped output projection:** the Q2_K CUDA kernel reads the existing
  quantized device tensor and reproduces the AVX-512 dot's four FMA
  accumulators and horizontal reduction order. The previous reduction's tiny
  error switched layer 4's sixth expert at decode position 33. After this
  fix all 718 shared intermediate dumps at that position matched bitwise.
  All groups execute in one kernel launch; no additional F32 VRAM weight mirror is allocated.
- **Same output head:** the graph head calls native `rmsnorm_proj` and
  `dense_proj` inside a scheduler node. This avoids GGML's Q8 activation
  rounding for its quantized LM-head multiplication.
- **Fixed route reduction:** DeepSeek collects unweighted per-route outputs
  and applies weighted FMA in k order in prefill and decode. Otherwise,
  summing per GPU changes the reduction tree when an expert changes cache
  location, and tiny roundoff can change later expert selection. Decode
  keeps one batched activation download per GPU, including CPU fallbacks in
  the same route order.
- **Both active-pool GPUs:** the former single active pool sent cold uploads
  through the 5060 Ti. Splitting the same total slot capacity lets the 3090
  serve more cold uploads. The controlled 48-token comparison improved from
  4.95 to 5.54 tok/s with identical token IDs, before the final reduction fix.

The output projection is enabled by default. Set
`ZEROMOE_DEEPSEEK_V41_OUTPROJ_GPU=0` to compare its native fallback.
Full-graph prefill and decode are the defaults.  The only remaining split is
the explicit `ZEROMOE_DEEPSEEK_V41_FULL_GRAPH_DECODE=0` native-decode
comparison; graph prefill stays enabled.

## Memory contract

RAMSafetyManager and GpuTopology are the sole owners of routed expert
weights. Graph nodes bind live slots, retain `hold_for_compute()` protection
through device work, and use the existing streaming path on a cache miss.
Graph construction, tokens, and phase transitions must not allocate a second
expert source, DDR arena, or VRAM weight pool.

Measured in the three-prompt graph run:

- Graph-owned MoE weight bytes: **zero**. Runtime guards reject a duplicate
  expert arena or source.
- Main expert caches: 2,555 slots across both GPUs; 32.735 GB resident at
  shutdown. Normal cache replacement remains managed by the existing tiers.
- DDR expert capacity: 8,360 slabs at 12,810,240 bytes each, 99.739 GiB maximum.
  The run filled 5,113 slots (65.499 GB); capacity is admitted on demand.
- Native dense pool: 844 tensors, 3.230 GB (about 3.01 GiB), shared with the
  graph. Graph private dense weight usage is zero; its backend marker arena
  is about 1 MiB.
- Decode graph: 434 nodes, one scheduler split, negligible device workspace.
  Attention, HC, Engram, MoE, and head nodes call native implementations and
  CUDA kernels. This is a full hybrid scheduler graph with host boundaries,
  not uninterrupted CUDA graph replay.
- Ordered MoE buffers hold activations only, approximately `topk * D` floats
  per device plus reusable host scratch. They never contain expert weights.
- Hold-for-compute failures and CPU expert fallbacks: zero. RSS reported at
  shutdown: 79.23 GB.

The existing seven-shard model passed GGUF verification: 3,344 tensors,
all 40 attention/norm layers, and Engram metadata complete. No model shard
was repacked, replaced, or deleted. The partial 4 KiB repack is unchanged;
expert stride alignment remains a possible later I/O optimization.

## Build and run

For the configured build used during this investigation:

```sh
cmake --build /tmp/zeromoe-v41-build --target zeromoe -j4
CUDA_VISIBLE_DEVICES=0,1 \
  /tmp/zeromoe-v41-build/zeromoe \
  --config configs/bench/deepseek_v41_100gib_ddr_17gib_vram_dual_gpu.toml
```

The standalone numerical regression test uses the real CPU AVX-512 dot and
GGML quantizer/dequantizer. It checks partial output blocks, padded group
strides, and the real `8 x 1024 x 4096` geometry on both GPUs. All 16,502
outputs matched bitwise, output guards stayed intact, and device weight bytes
were unchanged. The test passed on both the 3090 and 5060 Ti.

```sh
cmake --build /tmp/zeromoe-v41-build --target dsv41_grouped_outproj_test -j4
/tmp/zeromoe-v41-build/dsv41_grouped_outproj_test
GGML_SRC_DIR=/home/ray/Projects/llama.cpp-moestream-b11401/ggml/src \
GGML_LIB_DIR=/home/ray/Projects/ZeroMoE/build_b11401/ggml-build/src \
BIN=/tmp/zeromoe-v41-build/test_components make test-components
cmake --build /tmp/zeromoe-v41-build --target zeromoe_tests -j4
/tmp/zeromoe-v41-build/zeromoe_tests
```

`make test-components` passed all 45 component cases, the three route-policy
modes, the dynamic cache checks (1,017,845 checks, zero failures), and the
shared shape-pool check. A fresh `gguf_tool verify` passed for the model.

`make test` built the full native harness, then stopped at the existing EXL3
CUDA matmul/reconstruction/cache test. `ZEROMOE_EXLLAMAV3_SRC` is unset in
this build, and `cuda/exl3_aux.cu` therefore returns `cudaErrorNotSupported`
for EXL3 matmuls. This is not a full native-suite pass. The DeepSeek model
comparisons and dedicated grouped-projection test passed separately.

## Diagnostic artifacts

Temporary artifacts for this checkpoint:

- `/tmp/v41-ordered-three-graph.{stdout,stderr,logits}`
- `/tmp/v41-ordered-three-legacy.{stdout,stderr,logits}` (graph prefill)
- `/tmp/v41-ordered-three-native-full.{stdout,stderr}` (fully native)
- `/tmp/v41-exact-outproj-graph-p33/` and
  `/tmp/v41-native-head-legacy-p33/` (718 identical shared dumps)
- `/tmp/v41-grouped-outproj-test.log`
- `/tmp/v41-final-components.log`
- `/tmp/v41-final-native-tests.log`
- `/tmp/v41-final-model-verify.log`

GPU queries and inference require host device access in this environment.
A sandbox-only `nvidia-smi` failure does not establish that the NVIDIA driver
is down; an escalated query confirmed both adapters were healthy and idle.
