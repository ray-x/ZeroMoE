# Qwen3.6-35B-A3B UD-IQ2_M

## Status

The Qwen full-graph adapter builds with CUDA and passed the full-fit hardware
gate on an RTX 5060 Ti on 2026-10-26 using an 8-token smoke run. No 400-token
performance baseline is recorded yet.

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

The 8-token gate smoke passed `--full-fit --require-all-expert-layers-on-device`.
The planner reported `required_device_bytes=14007948866` and
`available_device_bytes=16338780160`; placement was 40/40 expert layers,
`graph_decode_runs=7`, and layered/legacy dispatch counters were zero. The
400-token run below remains the performance measurement.

For the performance measurement, run 400 generated tokens and preserve the
log with graph-use and all-expert placement evidence:

```sh
<CUDA_BUILD>/zeromoe --config configs/bench/qwen36_35b_a3b_iq2m_5060.toml --ngen 400
python3 tools/check_full_graph_moe_run.py logs/qwen36_35b_a3b_iq2m_5060.log \
  --full-fit --require-all-expert-layers-on-device
```

Record steady decode speed separately from prefill and first-graph setup.
Compare with llama.cpp using the same GGUF, context, prompt, generated-token
count, GPU, and thread count. No comparison number is frozen here yet.
