# Qwen3.6-35B-A3B UD-IQ2_M

## Status

The Qwen full-graph adapter builds with CUDA. The 5060 Ti full-fit request path
passes graph-prefill, graph-decode, and all-expert-placement gates. Performance
has varied across repeated runs: one run passed the 90% llama.cpp bar, while
the latest repeat reached 89.2%. Teacher-forced parity has moved past the
layer-3 RoPE defect, but broader numerical parity remains open: router inputs
drift by up to 7.81%, and six layers select a different near-boundary expert.
The model is not yet accepted as supported.

## GGUF architecture

The model file reports `general.architecture = qwen35moe`, which resolves to
the generic `Qwen` family. The adapter checks that exact architecture so its
registration does not imply support for other Qwen MoE architectures. The GGUF
metadata reports 40 transformer layers, 256 routed experts per layer, top-8
routing, a shared expert, and full attention every fourth layer; the other
layers use Gated DeltaNet.

The graph reuses the shared Qwen GDN, full-attention, routed-MoE, weight-arena,
and state-cache builders. Its block keeps Qwen's ordinary RMSNorm and residual
order, without the Qwen4Exp HC/PLE path. Qwen's GDN output gate uses SiLU.
The graph router matches the Qwen CPU path: softmax over experts, top-8, then
renormalization of selected weights; it does not apply group routing or
selection bias.
The adapter preflight checks required common, attention, GDN, expert, shared,
and output tensor shapes and supported quant-table types before reporting a
memory fit. An offline descriptor check against this GGUF found 732 required
tensors with no shape mismatch; startup's C++ preflight remains authoritative.

## Memory and placement contract

The planner sums all GGUF tensor bytes, one percent arena-alignment allowance,
F32 KV and GDN state sized to `engine.max_t`, and a 2 GiB decode-workspace
allowance. It queries free memory on the configured CUDA device. If the plan
fits, it selects every expert layer for the graph-owned device arena; partial
expert splitting and automatic layer clamping are disabled. The graph refuses
to run if any expert projection is not device-resident. A failed graph build
or decode is fatal under the full-fit policy.

For the 5060 Ti profile, an offline upper-bound estimate at a 4096-token
context is about 13.06 GiB, against 15.42 GiB free in one `nvidia-smi` sample.
That is only a planning estimate: startup's live memory plan and the graph's
placement log are authoritative.

The standalone `gguf_tool plan` is a read-only advisor. It reports dense tensor
requirements, expert slab sizing, detected hardware, and usage history; it does
not size graph KV/workspace or decide full graph placement. Use the engine's
startup memory plan and placement log for the full-graph fit decision.

Reproduce the read-only GGUF/system estimate without CMake:

```sh
./tools/run_gguf_tool.sh plan /path/to/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
```

Use the checked-in CUDA profile for graph testing; the read-only advisor does
not generate a cache profile or the graph adapter's runtime placement plan.
The graph-owned expert arena is the model's placement; the checked-in profile
keeps the legacy `moe_cache_gb` at zero to avoid reserving a duplicate expert
cache.

## 5060 Ti validation

The prepared profile is
[`qwen36_35b_a3b_iq2m_5060.toml`](../../configs/bench/qwen36_35b_a3b_iq2m_5060.toml).

Full-fit verification must pass
`--full-fit --require-all-expert-layers-on-device --require-full-graph-prefill`.
The 400-token benchmark passed with `graph_prefill_runs=4`,
`graph_decode_runs=1204`, 40/40 expert layers on-device, and zero
layered/legacy calls. The planner reported `required_device_bytes=14007948866`
and `available_device_bytes=16338780160`.

Reproduce the measured 16-prompt/400-decode benchmark and preserve its JSON:

```sh
CUDA_VISIBLE_DEVICES=0,1 <CUDA_BUILD>/zeromoe \
  --config configs/bench/qwen36_35b_a3b_iq2m_5060.toml --bench \
  --bench-random-pp 16 --bench-random-tg 400 --bench-reps 3 \
  --bench-warmup 1 --bench-seed 1 \
  --bench-out logs/qwen36_fullgraph_pg16_tg400_moe_fusion.json
python3 tools/check_full_graph_moe_run.py logs/qwen36_35b_a3b_iq2m_5060.log \
  --full-fit --require-all-expert-layers-on-device --require-full-graph-prefill
```

One three-repetition 400-token run measured a median of 112.39 tok/s
(8.896 ms/token p50, 8.932 ms/token p95), or 91.6% of the matching llama.cpp
benchmark at 122.73 tok/s. A later repeat on the pending changes measured
109.82 tok/s median (9.106 ms/token p50, 9.136 ms/token p95); its graph-only
counter was 109.44 tok/s, or 89.2% of llama.cpp, so that repeat failed the
default 90% performance gate. Both runs passed the full-fit graph execution
gate. This performance variation and the outstanding numerical-parity gaps
keep onboarding in progress. The passing-run ZeroMoE bench JSON is
`logs/qwen36_fullgraph_pg16_tg400_moe_fusion.json`; llama.cpp JSON is
`logs/qwen36_llama_bench_pg16_tg400_cuda1.json`.
The repeat artifacts are `logs/qwen36_fullgraph_pg16_tg400_repeat_20261026.json`
and `logs/qwen36_fullgraph_pg16_tg400_repeat_20261026.log`.

The main graph-runtime optimizations measured during this work were:

- GDN state snapshots now feed ggml-cuda's fused GDN-to-cache writeback. This
  removed 60 two-MiB device copies per decode replay.
- Qwen decode copies the strided convolution-history view directly into its
  cache; prefill still materializes contiguous history for the host handoff.
- The shared routed-MoE top-k reducer keeps its first expert as a view, allowing
  ggml-cuda to fuse the weighted expert reduction instead of launching one
  broadcast-add kernel per remaining route.
- Standard Qwen skips the unused final hidden-state D2H readback when graph
  logits are available and MTP or hidden-state diagnostics are disabled.

## Teacher-forced parity update

The latest saved comparison used the raw prompt `Tell a story.` (tokens
`39113, 264, 3255, 13`) and forced decode input token `146377` at position 4.
It compares the same GGUF through ZeroMoE and the llama.cpp CUDA trace. The
ZeroMoE trace contains four prompt rows followed by one decode row. Its files
are in `/tmp/qwen36_zero_gdn_decode_probe_rerun`; the llama.cpp reference is in
`/tmp/qwen36_llama_full_parity`.

The corrected four-lane IMRoPE and decode KV-cache packing remove the earlier
layer-3 RoPE divergence. At layer 3 prefill, Q/K after RoPE differ by
0.91%/0.94% relative RMS, attention context by 1.00%, and the output
projection by 1.02%. At decode, Q/K differ by 0.71%/0.89%, attention context
by 1.02%, and the output projection by 1.33%. Layer-0 GDN is especially
close: normalized Q/K differ
by 1.2e-6/3.1e-6, the GDN core by 4.2e-5, and its output projection by
0.0115% relative RMS.

The remaining drift accumulates through the 40-layer MoE stack. Router logits
differ by 0.37% mean and 0.69% maximum relative RMS; softmax probabilities
differ by 1.87% mean and 4.18% maximum. The MoE input differs by 3.92% mean,
reaching 7.81% at layer 39. The top-8 expert set matches in 34/40 layers;
layers 14, 17, 31, 33, 37, and 38 each replace one expert near the top-8
boundary, whose probability gap is below 3e-4 and as small as 9e-7. Among
layers with the same expert set, routed-MoE output differs by 4.96% mean and
10.78% maximum relative RMS. The six changed sets produce 14% to 31%
routed-output differences. This points to accumulated arithmetic differences
and top-k sensitivity, rather than another layer-3 RoPE or cache-layout bug.

The measurements reproduced in a fresh 2026-10-27 run. The full-graph gate
passed with one graph prefill, one graph decode, all 40 expert layers on the
device, and zero layered forwards or legacy expert dispatches. The gate log is
`/tmp/qwen36_parity_retry.log`; probe diagnostics were enabled, so its decode
timing is not a performance measurement. Earlier retries intermittently
reported `cudaGetDeviceCount == 0` or hit the existing-process preflight; the
successful run used no `--force-start`.

An earlier failed retry wrote preflight-only output to the profile's default
log path, replacing its prior contents. The benchmark JSON artifacts remain;
rerun the benchmark to regenerate the standard full-run log.
