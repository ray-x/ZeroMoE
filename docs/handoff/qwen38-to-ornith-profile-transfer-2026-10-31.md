# Qwen3.8 to Ornith profile transfer audit — 2026-10-31

## What was missed

The Qwen3.8 reference was used informally, but its measured run profile was not
frozen and diffed before the first long Ornith runs. The checked-in
configs/qwen38-flash.toml is a starter profile: it has a placeholder model path,
context 2048, and Q2_0 global top-K settings. It is not the measured benchmark
TOML.

The benchmark runner, bench/qwen_decode_profile.sh, defaults to
logs/graph_framework/merged-step2/correctness/early_1.toml and
logs/graph_framework/inputs/prompt.tokens. Both files are missing from this
worktree. The wrapper previously did not check either path; a missing TOML
could be replaced with an empty generated config. I added input and model-path
guards. Running the wrapper with its current missing defaults now exits before
launching an engine. The guard check is saved at
build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/qwen-profile-input-guard-check.txt.

We should have stopped at this missing-reference check, restored the measured
Qwen config/prompt inputs, or explicitly recorded a substitute reference. We
instead built and tuned custom Ornith profiles without a completed transfer
matrix. That caused late discovery of a prompt-specific usage ranking and the
early-ready async-output graph bug.

## Reference evidence

Qwen3.8's measured 3090 profiles are distinct:

- Layer split: 22 GPU / 26 CPU expert layers, 10 CPU threads, context 416,
  16 prompt / 400 generated, 28.66 tok/s.
- Q2_0 usage hotset: 14,554 experts, about 18.74 GiB, 10 CPU threads,
  43.12 tok/s.
- The benchmark runner sets CPU threads to 10, all-split early-ready on, and
  layer-13-only early-ready off. It leaves async output at the Qwen4Exp
  default.
- Qwen's usage file is model-specific and must not be reused for Ornith IQ3_XXS.

Reference sources are configs/qwen38-flash.toml,
docs/models/qwen38-flash.md, docs/perf-baselines.md, and
bench/qwen_decode_profile.sh. The measured benchmark TOML and prompt-token
file cited by that script are absent in this checkout.

## Setting-by-setting decisions

| Setting | Qwen3.8 reference | Ornith decision and evidence |
|---|---|---|
| GPU | RTX 3090 | Same card, pinned to UUID `GPU-fbddd113-32a1-56af-9c8e-7f2c877a9736` |
| Decode threads | Measured runner uses 10; starter TOML says 12 | Matched runs use 10 threads |
| Host source | 90 GiB host RAM, full preload, 45 GiB pinned budget, 10 fetch and 4 prefetch threads | Transfer these resource settings; use the host-level CUDA execution context |
| Graph policy | Required full graph for the measured path | Strict full-graph prefill/decode gates; strict memory preflight now rejects insufficient plans |
| Usage ranking | Qwen-specific Q2_0 usage file, global top-K 14,554 | Keep the Ornith five-prompt survey; rank by each expert's packed-byte cost |
| Expert capacity | About 18.74 GiB Q2_0 hotset | 17.0 GiB selects 5,561 experts at 81.58% survey coverage; 17.25 GiB selects 5,643 at 82.12% |
| Survey distribution | Measured prompt mix and TOML are absent from this checkout | Equal-weight five-prompt usage profile: 35.60 tok/s in prior sync-output run. An 8x benchmark-prompt weighting reached 42.50–42.54 but is experimental unless that mix matches expected use |
| Early-ready | All-split on; layer 13 off | Transfer both. Async CPU marker must be rooted; teacher-forced output replay passes 96 routed-output boundaries |
| Async output | Qwen4Exp default-on | Keep async on for `qwen4exp`; qwen35moe now defaults to sync output after the same-profile A/B |
| Context | Qwen starter TOML 2048; split benchmark 416 | Matched Ornith workload uses context 224 |
| Attention/state | Qwen4Exp HC/PLE plus its GDN/attention topology | Ornith qwen35moe has its own GDN/full-attention layout; do not transfer HC/PLE settings |

## Current evidence and next checks

- Async-marker root fix passes all 96 teacher-forced routed-output boundaries. The early-ready marker root is needed only in the async-output graph path.
- Matched output-copy runs: qwen35moe async-default 32.60 tok/s; explicit sync 34.46 and 34.06 tok/s. The new adapter-default sync run measured 35.55 tok/s. All full-graph gates passed; all missed 42 tok/s.
- The sync runs report producer GPU work pending in 182–192 of 192 early-ready lane calls; async-default reports zero. Summed CPU lane work is similar, while lane wait is higher with async output. This suggests an overlap difference; repeatable same-host A/B and per-layer GPU attribution remain open.
- The new default-sync binary source is `7d48e01de3ea2804ea6375b37858c23c909ad1d2`, SHA256 `0c6eaa2eaa38d3d5ea599e03fb8e23af5955793f32b45aabaf930478510677e7`, GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`; provenance passes. Default-sync r6 passes strict full graph at 17.0 GiB and reaches 35.55 tok/s.
- An 18.0 GiB budget does not fit: 23.97 GiB required, 23.28 GiB available. Strict preflight now refuses it before request generation. The 17.25 GiB batch-4 run fits at 23.22/23.28 GiB, passes graph gates, and reaches 34.71 tok/s; the 42 tok/s gate still fails.
- The 80% per-layer floor prototype passed full-graph gates twice and raised minimum layer coverage to 80.03%, but measured 33.70/33.53 tok/s versus 34.44 for the global hotness control under different CPU PSI. The runtime knob was removed; the offline selector simulation and lane timing remain as evidence.
- The runtime benchmark repeats the gradient-descent prompt three times. It is not a five-distinct-prompt runtime suite; keep the survey distribution and runtime workload distinct.
- The launch guard follows the Qwen documented rule for this 12-logical-CPU host: do not start above one-minute load 2. Record CPU PSI and process activity during preload and decode because USB loading can raise load average without equivalent CPU pressure.
- The isolated VNNI-on CPU candidate still fails routed-output parity at layer 0, even with custom cold fusion disabled; no performance run was made.

Use the reusable checklist at
[model-porting-profile-transfer-template.md](../design/model-porting-profile-transfer-template.md)
before another model's first long performance run, and preserve the resolved
settings matrix with its artifacts.
