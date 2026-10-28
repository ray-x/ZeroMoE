# Q2_0 versus UD-IQ1_M decode check, 2026-10-04

The claim that the simpler Q2_0 routed-expert arithmetic should be faster is
supported by isolated measurements on this machine. The earlier 31.76 versus
roughly 35–37 tok/s figures do not isolate quantization cost: they used
different prompts, continuations and measurement settings.

## Actual checkpoint types

Tensor headers were inspected from the two Qwen3.8-Flash-Next checkpoints:

| Tensor family | Q2_0 checkpoint | UD-IQ1_M checkpoint |
| --- | --- | --- |
| Routed gate/up, 48 layers | Q2_0 throughout | IQ1_M in 24 layers, IQ2_XXS in 24 |
| Routed down, 48 layers | Q2_0 | IQ4_NL |
| HC attention/FFN down/up | BF16 | Q8_0 |
| Router `ffn_gate_inp` | BF16 | F32 |

Both have routed gate/up shape `[2560,640,512]` and down shape
`[640,2560,512]`. HC down is `[10240,320]`, up is `[320,10240]`.
Other dense/shared/attention tensors also differ; neither checkpoint name
describes all its tensors. Census:
`logs/quant_compare_tensor_census_20261004.json`.

## Isolated routed-expert cost

[`bench/quant_route_cost.cpp`](../bench/quant_route_cost.cpp) builds the actual
decode-shaped GGML gate/up → SwiGLU → down graph, D=2560, FF=640, top-k=10,
with controlled valid route counts and separate gate/up tensors. Each
signature uses valid synthetic quantized weights. Routes rotate over 32
expert addresses; 20 warmup executions precede 120 timed executions. The
CPU uses 10 threads. CUDA uses graph replay on the RTX 3090 at the existing
220 W power cap. Times include backend submission and synchronization.

| Signature | CPU, 2 valid routes | CUDA, 8 valid routes |
| --- | ---: | ---: |
| Q2_0 / Q2_0 | 179.03 µs | 32.65 µs |
| IQ1_M / IQ4_NL | 195.94 µs | 45.25 µs |
| IQ2_XXS / IQ4_NL | 207.83 µs | 46.81 µs |

Q2_0 is faster for these controlled routed-expert cases on both backends.
This is not a whole-model throughput measurement. Logs:
`logs/quant_route_cost_{cpu,cuda}_20261004.log`.

## A real cost in the opposite direction: dense HC weights

[`bench/quant_hc_cost.cpp`](../bench/quant_hc_cost.cpp) measures eight fixed
HC down/up pairs in one CUDA graph, giving a working set larger than L2
without changing weight pointers or forcing capture per iteration. It uses
the actual shapes above, FP32 activations, valid synthetic BF16/Q8_0 weights,
20 warmups and 160 timed executions.

| HC pair type | Mean time per pair | Weight bytes per pair |
| --- | ---: | ---: |
| BF16 (Q2 checkpoint) | 29.07 µs | 13,107,200 |
| Q8_0 (UD checkpoint) | 17.11 µs | 6,963,200 |

There are 96 HC pairs in the 48-layer trunk (attention and FFN per layer).
Multiplying this isolated difference by 96 gives approximately 1.15 ms/token
of additional HC work for Q2_0. That is an estimate; whole-model cache and
scheduling behavior can differ. It partially offsets the faster routed
experts and shows why expert bit width alone cannot predict total decode.
Log: `logs/quant_hc_cost_cuda_20261004.log`.

## Matched whole-model and route checks

[`bench/quant_decode_compare.py`](../bench/quant_decode_compare.py) uses a
shared config derived from the Q2_0 cov80 profile; only the model path and
log path vary. Both models use 44 split layers, the same decode usage file,
80% right-sizing, context 2048, 90 GB RAM budget and 10 CPU threads. The
captured Q2_0 prompt IDs and 64-token continuation are replayed by both;
sampling remains enabled so its cost is retained. Six repeats alternate
Q2/IQ1 and IQ1/Q2 order. The warmup outputs were inspected for coherence.

Prompt: "Explain briefly the key concepts of neural networks and gradient
descent in plain English." Artifacts, effective configs, tokens, binary hash
and per-run timings: `logs/quant_decode_compare_chat_20261004/`.

The run had large time variation: Q2_0 ranged 23.40–31.56 tok/s, UD-IQ1_M
16.09–30.97. A background `aria2c` download and kernel-worker load were
observed during this series. The raw six-run means are 28.77 and 27.08,
respectively, but **this series does not establish a reliable baseline or
an accepted speed gain**. All repeats, including the slow ones, are retained.

Separate observer runs enable the engine's router tap and replay the same
history. Their timing is not used for throughput. Actual split-hot coverage:

| Model | GPU hot hits / 440 split routes per token | Hot rate | CPU routes, including 4 plain layers |
| --- | ---: | ---: | ---: |
| Q2_0 | 336.27 | 76.42% | 143.73 |
| UD-IQ1_M | 337.53 | 76.71% | 142.47 |

The near-equal counts do not explain a large quantization-specific gap on
this history. They also differ from the approximately 90% hot rate of the
historical hello-world benchmark. The usage survey config names UD-IQ1_M;
using that one survey across quantizations is not proof that real routing
coverage is identical. Route artifacts: `*_hits.engine.log`,
`*_hits.stdout`, `route_summary.json` in the matched-run directory.

An earlier hello-world/64-token series is excluded: Q2_0 stopped after a
short answer and `IGNORE_STOP=1` continued into artificial role/reasoning
text. Those timings are not representative of the requested reply. Its
logs remain in `logs/quant_decode_compare_20261004/`. The runner now rejects
warmup text containing a displayed stop marker and defaults to the longer
explanation prompt.

## Reproduction and remaining work

Use the same upstream integration GGML libraries as the engine. The route
probe links `-lggml-cpu -lggml-cuda -lggml-base`; the HC probe links
`-lggml-cuda -lggml-base`, with includes from
`../llama.cpp-moestream-integration/ggml/include` and library/rpath directories
`../moestream-build-upstream-master/ggml-build/src` and its `ggml-cuda` child.
Check competing GPU and CPU work before launching either CUDA probe.

The full-model comparison can be repeated in a quiet resource window:

```bash
python3 bench/quant_decode_compare.py \
  --engine logs/option_b_validation_engine_20261003/zeromoe \
  --out-dir logs/quant_decode_compare_quiet_new \
  --threads 10 --predict 64 --reps 6 --teacher-force --history Q2
```

The tested engine hash is
`517da98d1c8a41f0850fc112160a7325b4b5bdb113d565a5f9d4a974ede569c7`.
No engine/runtime code was changed for these measurements. Model verification
passed; new Python harness syntax and finite synthetic outputs were checked.

The linked GGML CUDA library has SM75 cubins plus compute75 PTX, verified in
its compile flags and with `cuobjdump`. This differs from the documented SM86
recipe for the RTX 3090. Its performance impact has not been measured; an
isolated SM86 build is a separate control, not an established explanation
for this quantization comparison. Do not replace shared libraries during a
matched run.

## Opt-in BF16 cuBLAS matvec experiment

The GGML integration source has a small experiment, saved as
[`patches/ggml-bf16-mmvf-cublas-experiment.patch`](../patches/ggml-bf16-mmvf-cublas-experiment.patch).
Set `GGML_CUDA_BF16_MMVF_CUBLAS=1` to send single-token BF16 matvecs to GGML's
existing cuBLAS BF16-input / FP32-accumulator path. It leaves explicit
`GGML_PREC_F32` requests on MMVF. The default remains MMVF.

Four interleaved HC microbench runs per path, with valid synthetic weights,
found:

| GPU | MMVF | cuBLAS BF16 | Change |
| --- | ---: | ---: | ---: |
| RTX 3090 | 29.09 µs/pair | 28.89 µs/pair | -0.7% |
| RTX 5060 Ti | 48.17 µs/pair | 40.99 µs/pair | -14.9% |

The pair is the HC down/up projections. It retains FP32 output and accumulation
while cuBLAS rounds the FP32 activation input to BF16. Against MMVF, synthetic
HC outputs differed by 0.24% RMS relative to output RMS. In the model-level
3090 check, three matched Q2_0 runs measured 31.28 tok/s (MMVF) and 30.95
tok/s (cuBLAS). A teacher-forced 64-step check had 98.44% top-1 agreement,
mean KL 0.000814, and mean teacher-token NLL increase 0.00233. It changed one
top-1 choice at a reference margin of 0.0109. The candidate's warm free-running
answer was coherent, but this is too little evidence to enable it by default;
the 3090 did not gain speed. The 5060 Ti result is a microbench only, not a
whole-model result.

Rebuild `ggml-cuda` after applying the patch. The flag is inherited by the
engine at process launch. The GGML CUDA library in these measurements was
built for SM75/PTX75, so an SM86 rebuild remains needed before treating these
timings as final 3090 kernel numbers.

The opt-in path was also measured in full Q2_0 `0/44/4` decode on the RTX
3090. Three runs of each arm used the same saved prompt IDs and the same
64-token continuation, 10 CPU threads, context 2048, identical Q2_0 config
and engine binary. MMVF averaged **31.28 tok/s**; cuBLAS BF16 averaged
**30.84 tok/s** (-1.4%). Both free-running warmup answers were coherent.
The teacher-forced raw logits agreed on top-1 for 63/64 steps, with mean
KL 0.000814 and mean teacher-token NLL increase 0.00233. The one top-1 change
was at a reference margin of 0.0109. Thus the 3090's microbench gain does
not carry over to decode throughput, and the BF16 input rounding can change
a token choice. Leave the flag off by default on this device.

The 5060 Ti has only the isolated HC microbench result above. No full-model
throughput or free-running quality result has been collected for that card;
the option remains experimental there too.

The full native Linux/CUDA suite passed after rebuilding with the patched
GGML library (`logs/ggml_bf16_experiment_native_test_20261004.log`). The
patch is also available as a standalone diff for the upstream integration
checkout. From its root:

```bash
git apply /path/to/ZeroMoE/patches/ggml-bf16-mmvf-cublas-experiment.patch
cmake --build /path/to/moestream-build-upstream-master --target ggml-cuda -j4
GGML_CUDA_BF16_MMVF_CUBLAS=1 CUDA_VISIBLE_DEVICES=1 ./path/to/zeromoe --config path/to/q2_config.toml
```

Use `CUDA_VISIBLE_DEVICES=0` for the 3090. The explicit FP32 precision path
is preserved. Leave the flag unset for the normal MMVF path.
