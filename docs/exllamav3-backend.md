# ExLlamaV3 execution backend

The `exllamav3` backend delegates the complete EXL3 inference pipeline to the
local ExLlamaV3 checkout: model construction, attention/GDN, HyperConnections,
KV and recurrent caches, routing, GPU experts, CPU experts, and generation.
MoEStream's native EXL3 kernels are not called on this path. The CLI replaces
itself with the configured Python process, using argv rather than a shell.

To enable the upstream backend, use an EXL3 TOML profile with
`model_format = "exl3"`, `exl3_backend = "exllamav3"`, the upstream source
directory, the isolated Python executable, and the desired CPU expert-layer
boundary. The checked-in 4bpw 26/22 and 2.05bpw 44/4 and 26/22 profiles are
working examples;
the `README.md` EXL3 section links to this guide.

The minimal backend-specific settings are:

```toml
model_format = "exl3"
exl3_backend = "exllamav3"
exllamav3_source_dir = "/home/ray/Projects/exllamav3"
exllamav3_python = "../.exl3-runtime/bin/python"
exllamav3_gpu_memory_gb = [22.0, 14.0]
exl3_gpu_expert_layer_begin = 26
omp_max_threads = 6
exllamav3_cuda_schedule = "blocking"
```

Set `model_path` to the safetensors checkpoint and supply metadata/tokenizer
paths when the model folder lacks the Hugging Face config files. Context,
generation length, prompt, and sampling options remain ordinary MoEStream
config entries. The profiles in `configs/` provide complete, tested examples.

On the Xeon W-2235 with the RTX 3090 and RTX 5060 Ti, six CPU-worker threads
are the measured default. Twelve threads contend with GPU submission and
nearly halve decode throughput.

The `exllamav3_cuda_schedule` setting selects the CUDA host wait mode
(`auto`, `spin`, `yield`, or `blocking`). It preserves unrelated CUDA device
flags and applies before loading the model. On this host, `blocking` is about
1.8% faster than `auto`.

The checkpoint includes an MTP draft head. Benchmark it with `--mtp`; the
report records its added load time and accepted draft-token count. Baseline
throughput does not use draft tokens.

`configs/qwen4exp_exllamav3_26_22.toml` requests the first 26 routed-expert
layers on ExLlamaV3's CPU worker and the remaining 22 on GPU. Actual per-layer
placement is checked after loading; ineligible offload silently falling back
to GPU is rejected. Dense/attention work remains on GPU. ExLlamaV3 autosplits
GPU modules under the configured per-device memory budgets, so the division
between the two GPUs need not match the native backend's 15/7 GPU allocation.

The CPU expert implementation is upstream `cpu/moe_mul1.cpp` (persistent
worker pool, runtime AVX2/AVX-512/VNNI dispatch). Upstream
`parallel/all_reduce_cpu_avx512.cpp` handles tensor-parallel communication,
not the expert matmuls. Whole-layer CPU offload requires layer-split mode,
not tensor-parallel mode.

## Environment

An isolated environment is used; no global Python installation or upstream
source modification is required:

```bash
uv venv --python 3.12 .exl3-runtime
uv pip install --python .exl3-runtime/bin/python torch --torch-backend cu130
uv pip install --python .exl3-runtime/bin/python \
    -r /home/ray/Projects/exllamav3/requirements.txt

export PYTHONPATH=/home/ray/Projects/exllamav3
export TORCH_EXTENSIONS_DIR="$PWD/.exl3-runtime/torch_extensions"
export TORCH_CUDA_ARCH_LIST='8.6;12.0'
export MAX_JOBS=4
export CUDAHOSTCXX=/usr/bin/g++-15
```

Use the installed host compiler and CUDA flavor appropriate to your machine.
The extension builds on first import. Keep the same extension directory and
architecture list for subsequent launches to reuse that build.
The supplied config sets this extension directory, architecture list, and
CUDA host compiler when they are not already overridden in the environment.

## Checkpoint metadata

The local checkpoint has all nine safetensors shards, the n-gram table, and
`tokenizer.json`, but no HF `config.json`. The upstream backend requires HF
metadata rather than the native decoder's GGUF metadata substitute.

The folder's `exl3_3bpw` name is not its tensor bitrate: independent header
inspection and `exl3_tool verify` find **75,277 4-bit matrices, two 5-bit
matrices, and 308 6-bit matrices**. For example, a routed expert trellis has
shape `[160, 40, 64]`, hence `64 / 16 = 4` bits/weight. The runner records
actual tensor bitrates in its benchmark JSON rather than inferring them from
the directory name. See `logs/exllamav3_runtime/verified_checkpoint.log` and
`logs/exllamav3_runtime/verified_rates.log`.

The validation metadata is retrieved from the official
[Qwen configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/21a7c3ca68484a9987d62fd51b78130d9b9a11f8/config.json)
at revision `21a7c3ca68484a9987d62fd51b78130d9b9a11f8`:

```bash
mkdir -p logs/exllamav3_runtime/metadata
for name in config.json tokenizer_config.json; do
    curl -fL "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/21a7c3ca68484a9987d62fd51b78130d9b9a11f8/$name" \
        -o "logs/exllamav3_runtime/metadata/$name"
done
```

The adapter creates a temporary directory of symlinks combining these small
metadata files with the existing checkpoint. It neither copies nor modifies
model weights, accepts symlinked source shards, rejects broken links, and
removes the view after worker shutdown. Relative paths in this backend's
config are interpreted relative to the TOML file, not the caller's directory.

## Run and benchmark

Validate configuration and model-view paths without importing CUDA:

```bash
python3 tools/run_exllamav3.py \
    --config configs/qwen4exp_exllamav3_26_22.toml --check
```

This does **not** prove tensor completeness, successful loading, or inference
correctness. Those require the upstream runtime to load and execute the model.

After rebuilding MoEStream, normal inference can use the same config entrypoint:

```bash
./build_ggmlcuda/moestream \
    --config configs/qwen4exp_exllamav3_26_22.toml --prompt "Hello" --ngen 8
```

For a fixed-length, greedy benchmark without early EOS stopping:

```bash
.exl3-runtime/bin/python tools/run_exllamav3.py \
    --config configs/qwen4exp_exllamav3_26_22.toml \
    --benchmark --warmup-tokens 8 \
    --output-json logs/exllamav3_runtime/benchmark_26_22.json
```

The benchmark uses upstream `Generator`/`Job` timings, separates loading and
prefill from decode, and reports `(generated_tokens - 1) / time_generate` to
match the native 128-token/127-interval comparison. The JSON includes token
IDs, completion text, source revision, PyTorch version, and observed layer
placement. Benchmark mode omits per-token console output. Normal inference
honors upstream EOS tokens. Both modes reject existing GPU compute processes
and hold MoEStream's shared host-admission lock throughout loading and inference.

## Measured results

On 2026-10-05, fixed 128-token runs used the same prompt and verified 26 CPU /
22 GPU layer placement:

| CPU worker threads | CUDA wait | Mode | Decode tok/s | Decode time |
| ---: | :--- | :--- | ---: | ---: |
| 12 | Auto | Fixed output | 15.99 | 7.944 s |
| 6 | Auto | Fixed output | 26.39 | 4.813 s |
| 6 | Blocking | Fixed output | 26.86 | 4.728 s |
| 6 | Blocking | MoEStream CLI, streamed output | 26.61 | 4.773 s |
| 7 | Blocking | Fixed output | 24.68 | 5.147 s |
| 5 | Auto | Fixed output | 25.98 | 4.888 s |
| 4 | Auto | Fixed output | 22.44 | 5.661 s |

The six-thread run is a 65% increase over twelve threads. Those two runs
produced identical prompt and generated token IDs across all 128 tokens. The
six-thread/blocking profile measures 26.61 tok/s through the MoEStream CLI.
Fixed-length output measures 26.86 tok/s. The 28 tok/s target remains unmet.
The upstream worker reports `avx512-vnni`; this Xeon
does not expose `avx512_vbmi`. Profiling shows most CPU expert time in gate/up
GEMV, followed by down GEMV. Setting upstream's `EXL3_MOE_CPU_SMALL_WORKERS=2`
reached 26.30 tok/s, so leave it unset. The optional MTP run accepted 68 draft
tokens and rejected 172 (28.3% acceptance), reducing decode to 16.51 tok/s.
See the individual logs and JSON under `logs/exllamav3_runtime/`.

The full-checkpoint runs verify loading, coherent output, and the requested
26/22 placement. CPU experts stay RAM-resident in the upstream worker rather
than using MoEStream's native on-demand CPU LRU. The 28 tok/s target remains
unmet; the measured peak is 26.86 tok/s. MTP is opt-in and measured slower.

## User-supplied 2.05bpw checkpoint

The requested `/mnt/usb-ssd/Qwen3.8_flash/exl3_2bpw` model passes
`exl3_tool verify`: five shards, 48 layers, 512 experts, payload smoke passed;
its actual trellis rates are 75,277 2bpw matrices, two 3bpw, and 472 4bpw.
This is a real 2.05bpw pack and is distinct from the earlier `exl3_3bpw`
directory, whose matrices were almost all 4bpw. The removable-drive model
folder contains no `tokenizer.json`, so the profiles symlink the matching
Qwen tokenizer from the existing 4bpw directory through
`exllamav3_tokenizer_path`; model weights still come from the 2.05bpw shards.
`exl3_tool verify` checked every tensor range and payload smoke test; its
record is `logs/exllamav3_runtime/exl3_2bpw_verify.log`.

Three verified-config profiles are available:

| Profile | CPU routed layers | GPU routed layers |
| --- | ---: | ---: |
| `configs/qwen4exp_exllamav3_2bpw_0_44_4.toml` | 44 | 4 |
| `configs/qwen4exp_exllamav3_2bpw_0_43_5.toml` | 43 | 5 |
| `configs/qwen4exp_exllamav3_2bpw_26_22.toml` | 26 | 22 |

For ExLlamaV3 whole-layer offload, these names mean layers `0..43` or `0..42`
run routed experts on the CPU worker, and the tail four or five layers run them
on GPU. Autosplit then places actual modules over the configured 3090/5060 Ti
budgets. I ran each profile's CPU-only `--check`; it verifies the checkpoint,
tokenizer and requested layer count, not CUDA loading or VRAM fit.

Verify the pack, then run the first profile on an otherwise idle host:

```bash
./build_ggmlcuda/exl3_tool verify /mnt/usb-ssd/Qwen3.8_flash/exl3_2bpw
./build_ggmlcuda/moestream --config configs/qwen4exp_exllamav3_2bpw_0_44_4.toml \
    --ngen 128
```

The fixed-length 0/44/4 benchmark has now loaded and generated 128 tokens:

| CPU routed layers | GPU routed layers | Load | Prefill | Decode | Placement/result |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 44 | 4 | 88.68 s | 1.496 s | 22.99 tok/s | Layers 0–43 on CPU; layers 44–47 on `cuda:0`; fit succeeded |

This run used six CPU worker threads, blocking CUDA waits, eight warmup tokens,
the same 42-token garden prompt, and fixed 128-token generation. The four GPU
expert layers fit on the RTX 3090; 15,306 MiB (about 14.95 GiB) remained free
after load. Although
the 5060 Ti was available, autosplit placed these four routed-expert layers on
`cuda:0`. Since 0/44/4 fit, the conditional 0/43/5 fallback was not run. The
22.99 tok/s result is below the requested 25 tok/s floor and is not directly
comparable to the 26/22 result above because the CPU/GPU expert-layer split is
different. Full output, placement, and timing are in
`logs/exllamav3_runtime/benchmark_2bpw_0_44_4.log` and
`logs/exllamav3_runtime/benchmark_2bpw_0_44_4.json`.

For a repeatable fixed-length run of this profile, use the same runner as the
26/22 benchmark:

```bash
.exl3-runtime/bin/python tools/run_exllamav3.py \
    --config configs/qwen4exp_exllamav3_2bpw_0_44_4.toml \
    --benchmark --warmup-tokens 8 \
    --output-json logs/exllamav3_runtime/benchmark_2bpw_0_44_4.json
```

The 2.05bpw pack was also tested at the target 26-CPU/22-GPU split, using the
same prompt, six CPU threads, blocking CUDA waits, eight warmup tokens, and
fixed 128-token output:

| Trial | CPU routed layers | GPU routed layers | Load | Prefill | Decode |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | 26 | 22 | 88.83 s | 0.977 s | 31.23 tok/s |
| 2 | 26 | 22 | 90.61 s | 0.970 s | 31.37 tok/s |
| Mean | 26 | 22 | — | — | 31.30 tok/s |

Both runs exceed 28 tok/s. Routed experts in layers 26–47 all landed on
`cuda:0` (RTX 3090); the 5060 Ti received no routed-expert layers. Prompt
token IDs match across runs; generated IDs match for the first 119 of 128
tokens and diverge in the final nine. Both completions remain coherent. Full
placement, token IDs, and timings are in
`logs/exllamav3_runtime/benchmark_2bpw_26_22.json` and
`logs/exllamav3_runtime/benchmark_2bpw_26_22_repeat.json`.

Repeat the fixed-length measurement with:

```bash
.exl3-runtime/bin/python tools/run_exllamav3.py \
    --config configs/qwen4exp_exllamav3_2bpw_26_22.toml \
    --benchmark --warmup-tokens 8 \
    --output-json logs/exllamav3_runtime/benchmark_2bpw_26_22_repeat.json
```

Q2_0 comparison work remains deferred until EXL3 performance tuning is
complete, as requested.

## Tests

```bash
python3 -m unittest discover -s tests -p test_exllamav3_backend.py -v
make test-components ARGS="exllamav3_backend_launch"
```

These cover configuration, symlinked views, cleanup, layer-placement checks,
prompt formatting, GPU-process rejection, and the no-shell C++ launch adapter.
The latest Python run on 2026-10-05 passed all 11 tests.
The runtime evidence above separately validates loading, actual routed-layer
placement, and fixed-length decode for the 4bpw 26/22 and 2.05bpw 44/4 and
26/22 profiles. Unit tests and config checks do not replace full-checkpoint
quality validation.

## Performance investigation

The local directory is named `exl3_3bpw`, but its trellis headers show 75,277
4bpw matrices, two 5bpw matrices, and 308 6bpw matrices. The official
[Turboderp EXL3 repository](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.05bpw_h5_ng5)
has a real 3.05bpw Qwen3.8 pack. The 4bpw checkpoint therefore sends about a
third more packed expert data through the RAM/CPU/GPU pipeline than a 3.05bpw
pack would, before scale and padding overhead.

The [ExLlamaV3 release benchmarks](https://github.com/turboderp-org/exllamav3/releases/tag/v1.5.0)
report 62.77 tok/s on Qwen3.8 at 3.05bpw using a 3090 Ti + 3090, an 8-core
AVX2 CPU, and 20% CPU offload with a pinned arena and the n-gram table in RAM.
That hardware, quant, offload fraction, and memory policy differ substantially
from this machine's 4bpw pack, six-core W-2235, 3090 + 5060 Ti, and 26 CPU /
22 GPU layer split; the number is context, not an apples-to-apples target.

I could not find an official ExLlamaV3/TabbyAPI result backing the quoted
"up to 2.5x faster than llama.cpp" prompt-processing figure. The official
release table instead compares ExLlamaV3 versions: its Qwen3.8 3.05bpw Pro
6000 row shows prefill rising from 5,788 to 7,327 tok/s (+27%), not 2.5x
versus llama.cpp. The separate 80%-CPU-offload row is on different hardware
and also compares release versions. These are prompt-prefill figures; this
profile used a 256-token context and a 42-token prompt, so our 26.61 tok/s
streamed CLI result measures decode and says nothing about 2,048-token
prompt processing. A valid 2,048-token comparison needs a matched prompt,
model quant, GPU set, offload split, prefill chunk, and timer boundary across
both backends.

VNNI is already running here, and the upstream [CPU expert tuning guide](https://github.com/turboderp-org/exllamav3/blob/master/doc/env_vars.md)
reports that swizzling improves its cold-expert GEMV bandwidth (+40% in its
7960X test). This runner leaves the upstream swizzle and CPU affinity defaults
enabled. The six-thread profile's own phase timings show gate/up GEMV as the
largest CPU phase, followed by down GEMV. Intel rates this W-2235 at four
DDR4-2933 channels and 93.85 GB/s theoretical memory bandwidth, but we have
not collected hardware DRAM counters to determine how much of that bandwidth
these EXL3 GEMVs sustain. The CPU worker also lacks AVX-512 VBMI, so this host
uses ExLlamaV3's AVX-512 VNNI tier.

The same upstream guide leaves the n-gram table streamed from disk by default
for SSD-class storage. This checkpoint's symlink resolves to a SATA SSD, so
loading the entire tens-of-gigabytes table into RAM is not yet justified as a
decode fix. The local MTP run accepted only 68 of 240 proposed draft tokens
(28.3%) and decoded more slowly, so MTP stays opt-in.

### 4bpw per-layer expert split (recommended 4bpw default)

Whole-layer 26/22 on the 4bpw pack peaks at 26.86 tok/s. Upstream's per-layer
expert split (`exllamav3_cpu_expert_split = 278`) runs the tail 278 routed
experts of every layer on CPU and 234 on GPU, a footprint roughly equal to
26/22, and overlaps CPU and GPU expert compute within each layer. It is
mutually exclusive with `exl3_gpu_expert_layer_begin`.

Upstream swaps hot and cold experts dynamically by default, which made decode
vary between 28.48 and 26.86 tok/s. Setting `exllamav3_cpu_expert_swap = false`
(the runner exports `EXL3_MOE_CPU_SWAP=0`) pins placement statically:

| Trial | Decode |
| :--- | ---: |
| 1 | 28.34 tok/s |
| 2 | 28.07 tok/s |
| 3 | 28.18 tok/s |
| Mean | ~28.20 tok/s |

All three trials clear 28 tok/s. This split with static placement is the
recommended 4bpw profile: `configs/qwen4exp_exllamav3_4bpw_cpu_split_278.toml`.
The key defaults to `true` (upstream dynamic swapping) when omitted.

The 2.05bpw 26/22 profile reaches a repeatable 31.30 tok/s mean and exceeds
the 28 tok/s objective. On the 4bpw pack, the post-graph blocking baseline was
26.67 tok/s; CUDA `spin` reached 26.26 tok/s, disabling CPU pinning reached
22.27 tok/s, and `EXL3_MOE_CPU_SMALL_WORKERS=1` reached 24.13 tok/s. The
two-worker small-job cap reached 26.30 tok/s. A diagnostic CPU profile
measured steady gate/up GEMV at 581 us/job and down GEMV at 290 us/job; it is
instrumented and its throughput is not a baseline. The historical 4bpw peak
remains 26.86 tok/s for whole-layer placement, 4.1% short of 28; the 4bpw
pack clears 28 only with the static per-layer expert split above. CPU expert gate/up GEMV remains the primary
optimization target for the 4bpw pack. Resume Q2_0 comparisons after the EXL3
performance work is complete.
