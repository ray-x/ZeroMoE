# Qwen full-graph prefill state handoff

Set this root-level TOML key alongside both full-graph switches:

```toml
qwen4exp_full_graph = true
qwen4exp_full_graph_decode = true
qwen4exp_prefill_state_handoff = true
```

The handoff defaults to **false**. The IQ2_XS RCO50 3090 GPU-only profile enables
it; the 5060 Ti profile leaves it disabled. The environment override
`ZEROMOE_QWEN4EXP_PREFILL_STATE_HANDOFF=0` or `1` takes precedence.

When enabled, completed prefill GDN state and convolution history stay in
VRAM. Fresh prefills read immutable GPU zero-state inputs, avoiding repeated
host clearing and state uploads. Decode binds its recurrent/PLE state to those exact output addresses.
The completed prompt graph is retained separately from the startup weight-arena
owner: startup graphs contain no completed prompt state. KV and PLE host
snapshots are still exported. Layered fallback materializes current state on
demand and retires the active decode graph rather than reusing stale state.
Prompt caching, MTP, compact expert staging, and explicit arena recreation keep
the host-copy path. Incompatible state storage also falls back to host copies.

The prefill scheduler workspace is **not reclaimed during decode** with this
option enabled: the latest pp23 graph on the 3090 retains about 0.175 GiB of
scheduler workspace, down from the initial 0.611 GiB. Reusable zero-state input
arenas also occupy about 0.111 GiB per retained prefill graph. Costs scale with
prompt shape. The host-copy path remains available by disabling the
option. Matching prompt shapes can reuse the retained prefill graph.

## Reproduce

The standard `build/zeromoe` binary is updated from the validated
`build_fork_diag` build. The prior binary is retained as
`build/zeromoe.before-state-handoff`. The updated binary resolves its ggml
libraries from `build_fork_diag`; keep that directory.

```bash
./build_fork_diag/gguf_tool verify /data/sda/models/qwen3.8-flash-next/GSQ-RCO-IQ2_XS_RCO50
./build/zeromoe --graph-device-expert-layers 48 --graph-device-expert-trailing 1 --config configs/testing/qwen4exp_gsq_rco_iq2xs_rco50_3090_gpu_only.toml
```

For matched warm measurements, the script runs six identical 23-token requests
per mode and discards the first. All 48 expert layers stay on GPU; no CPU expert
offload is configured. The checked-in token IDs preserve the exact prompt used
in the earlier Bee comparison. Override `ZEROMOE_HANDOFF_PROMPT_TOKENS` for a
different model/tokenizer.

```bash
bash tests/qwen4exp_state_handoff_ab.sh ./build/zeromoe
ZEROMOE_HANDOFF_CHECK_LOGITS=1 \
bash tests/qwen4exp_state_handoff_ab.sh ./build/zeromoe \
  configs/testing/qwen4exp_gsq_rco_iq2xs_rco50_3090_gpu_only.toml \
  logs/qwen4exp_state_handoff_correctness
```

Logit dumping adds overhead; use the first command for performance and the
second for bit-exact correctness. The script requests 129 output tokens so
there are exactly 128 decode forwards/output intervals after the first
prefill-produced token. Decode rates now use first-to-last output timing;
Qwen decode graph preparation precedes the first output and the final unused
forward is omitted. Rebuild the validated dependency selection with:

```bash
cmake -S . -B build_fork_diag -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DZEROMOE_GGML_ROOT="$PWD/.ggml-fork-diag/ggml"
cmake --build build_fork_diag -j10
```

## Historical measurements: 2026-10-03

Same fresh binary, RTX 3090, 23 identical prompt tokens, five samples after one
warmup, no logits/hidden dumps in timed runs:

| Metric | Host-copy mode | Device handoff |
| --- | ---: | ---: |
| Warm prefill-only wall time | 132.465 ms | 91.954 ms |
| Warm prefill-only throughput | 173.63 tok/s | 250.12 tok/s |
| State-cache host export | 39.590 ms | 0.698 ms |
| Scheduler submit + synchronization | 74.316 ms | 73.101 ms |
| Time outside the executor's compute ledger | 18.504 ms | 18.102 ms |
| Prefill in combined pp23/tg128 requests | 178.83 tok/s | 246.41 tok/s |
| Decode graph preparation in combined requests | 109.20 ms | 10.82 ms |
| Decode wall rate, **including setup** | 43.68 tok/s | 45.43 tok/s |

The earlier ~49 ms export measured a stale binary. The freshly rebuilt baseline
measures ~39.6 ms; the handoff removes that work, not just its timer. The earlier
Bee pp23 reference is ~368 tok/s, so the remaining prefill gap is **not fixed**:
about 18 ms outside the compute ledger plus the remaining execution difference
still need investigation. These are not all GPU-kernel timings.

Artifacts are in `logs/qwen4exp_state_handoff_ab_final/`. Separate correctness
runs compare all 128 decode-logit vectors and the prefill hidden row bit-for-bit
against the host-copy baseline. Changing prompt shapes (8 then 7 tokens) and
forced decode KV growth (eight graph preparations) also produce bit-identical
logits. GGUF verification and the focused config parser/sample-config test pass.

The initial native suite run failed in expert-upload Test 23 and KDA Test 30.
The refreshed 2026-10-04 current-tree native run passes all core tests; details
are in `logs/qwen4exp_bee_fullgpu_compare/native_tests_current.log`.

## Final matched comparison: 2026-10-04

The standard binary is updated to the verified fast path. On RTX 3090, all
48 expert layers and all three projections stay resident in CUDA (16.94 GiB
of expert weights). Decode has 6,512 CUDA nodes, zero CPU nodes, one scheduler
split, and CUDA graph replay enabled. Mixed CPU/GPU profiles preserve their
existing staging topology is selected automatically for mixed host/device
placement. The former `ZEROMOE_QWEN4EXP_GRAPH_MATCH_LLAMA_CPP_TOPOLOGY`
force override was removed on 2026-10-28;
`ZEROMOE_QWEN4EXP_PREFILL_DEVICE_STATE=0` restores host zero-state inputs.

Five warm repetitions, same 23 prefill IDs, identical 128 decode inputs, no
logit dumping during performance measurement:

| Phase | Bee | ZeroMoE |
| --- | ---: | ---: |
| Prefill / pp23 | 409.68 tok/s | 431.25 tok/s |
| Decode / tg128 | 70.52 tok/s | 69.21 tok/s |

The current-tree rerun has prefill 5.3% faster; decode differs by **1.86%**,
meeting the <3% target. The earlier isolated run measured 361.81/375.82 tok/s
prefill and 62.18/61.31 tok/s decode (1.40% difference); use paired results,
not numbers taken from different runs.
Both prefills include the output projection. ZeroMoE uses 129 emitted tokens
to measure 128 first-to-last intervals, excludes setup before the first output,
and performs no unused forward after the last output. Teacher forcing skips
sampling so the comparison matches Bee's inference-only benchmark.

The final reproduction script waits for competing inference rather than
interrupting it:

```bash
bash tests/qwen4exp_bee_fullgpu_compare.sh
```

The workspace Bee benchmark in
`logs/qwen4exp_perf_gap/bee_forced/llama-bench` links the existing instrumented
Bee libraries. Its additions provide an exact decode-token file and isolated
prefill/decode phase timers; they do not change model execution or ggml kernels.
The reproducible source patch is
`patches/bee-bench-exact-decode-and-phase-timing.patch`; the external Bee source
tree is left unchanged. Artifacts for the final gate are in
`logs/qwen4exp_bee_fullgpu_compare_current/`, with the earlier run preserved in
`logs/qwen4exp_bee_fullgpu_compare/`.

The latest warm executor ledger is approximately 48.67 ms submission, 10.08 ms
synchronization, 0.23 ms output retrieval, 0.71 ms remaining state export, and
0.67 ms outside that ledger. The original ~18 ms host initialization plus
~114 MiB per-prompt state upload are removed rather than hidden in another timer.

Host-copy versus device-state modes produce bit-identical prefill hidden rows
and all 129 output-logit vectors in the final A/B. Switching the *first* output
projection from CPU to standard quantized GGML CUDA changes that vector's
rounding/activation quantization (max absolute delta 0.1066 against the prior CPU
path for the recorded prompt); its selected token is unchanged. The remaining
127 vectors compared against the previous 128-output run are bit-identical.
This is not a claim of bit-exact logits against Bee. Arithmetic, Python code,
and a short factual prompt pass smoke tests; all core native tests pass.
