# Ornith-1.5-122B drift investigation — 2026-10-30

> Superseded by the [2026-10-31 recovery update](ornith-122b-recovery-2026-10-31.md).
> This file preserves the earlier first-divergence investigation; its next-step
> and paused-work statements are historical.

## State

Ornith is **not correctness-accepted or performance-accepted**. This round isolated the first numerical difference to the routed expert FFN output in layer 0. Disabling the IQ1_S VNNI path did not remove it, and matching ZeroMoE's CPU build mode to llama.cpp's did not remove it either. The exact low-level operation that produces the difference remains unknown.

No performance tuning was done in this round. The two-token teacher-forced run is an instrumented correctness repro, not a throughput measurement.

## Reproducible comparison

- Model: `/mnt/usb-ssd/@/home/ray/models/Ornith-1.5-122B-UD-IQ3_XXS.gguf` (49,515,667,712 bytes).
- GPU: RTX 3090 only, UUID `GPU-fbddd113-32a1-56af-9c8e-7f2c877a9736`.
- Workload: `Hello there.`, prompt tokens `9419, 1017, 13`, forced decode token `353`, context 128, 10 threads.
- Placement: all 48 MoE layers on CPU/RAM; dense/state graph on CUDA; full graph planner reported `Fits`.
- Reference source: llama.cpp `8baabd53b77d1c48c6ef8d3b620436292f3f6231`, clean worktree.
- CPU build options matched for the comparison: `GGML_NATIVE=ON`, `GGML_AVX2=OFF`, `GGML_AVX512=OFF`, `GGML_AVX512_VNNI=OFF`; generated CPU compile command uses `-march=native`.
- ZeroMoE runtime provenance passed for source `7d48e01de3ea2804ea6375b37858c23c909ad1d2`, dirty snapshot, GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`. `ldd` confirms the matched CPU `ggml`/`ggml-base`/`ggml-cpu` libraries and the expected ZeroMoE CUDA backend were loaded.

Artifacts: [`same-placement-parity-cpu-match`](../../build_mtp_results/2026-10-30-ornith-122b-same-placement-3090/iq1s-vnni-off/same-placement-parity-cpu-match/), including the run script, config, logs, `ldd`, provenance, and tensor probes.

## Evidence

The full-graph run gate passed and the prompt token IDs matched. The boundary comparator stopped at the first mismatch, pass 0 layer 1:

- Layer 0 input, router probabilities, selected weights, and expert IDs matched exactly.
- Layer 0 routed output (`llama.cpp ffn_moe_out-0` vs ZeroMoE `L0.moe_routed`, first three real token rows): max absolute difference `1.74622983e-10`, mean `1.05946995e-11`, 4,418 of 9,216 values unequal.
- Layer 0 shared branch matched bitwise. The combined MoE output differed by at most `4.65661287e-10`.
- At layer 1, hidden-state max difference was `2.38418579e-7`; router-probability max difference `2.23517418e-8`; normalized-weight max difference `2.98023224e-8`; ordered expert IDs still matched.

The small layer-0 routed-branch difference propagates into later hidden states. A prior route-only report from a CPU-build-mismatched run showed later expert-ID changes near router ties; repeat that analysis after isolating this branch, because that older run is provisional.

The comparator report and compact repro are in [`comparison-report.md`](../../build_mtp_results/2026-10-30-ornith-122b-same-placement-3090/iq1s-vnni-off/same-placement-parity-cpu-match/comparison-report.md) and [`first-divergence/pass0-layer1`](../../build_mtp_results/2026-10-30-ornith-122b-same-placement-3090/iq1s-vnni-off/same-placement-parity-cpu-match/first-divergence/pass0-layer1/).

## Ruled out and invalid comparisons

- Turning off the patched IQ1_S AVX512-VNNI path produced the same first mismatch and the same layer-0 routed-output delta. That patch is not the cause of this first divergence.
- The CPU compiler-mode mismatch also was not the cause: after rebuilding ZeroMoE's CPU GGML with the reference's `-march=native` settings, the same layer-0 difference remained.
- A clean standalone CPU GGML library mixed with the existing llama.cpp CUDA backend failed backend discovery (`Available devices: (none)`, backend init symbol not found). It did not load the model and is not a correctness result. Keep the library stacks coherent for future comparisons.
- The first matched-CPU run completed inference and dumped 1,360 probe files, but its wrapper then failed because its repository-root path was one directory too high. The full-graph gate and boundary comparator were run manually afterward and passed/failed as recorded above. The wrapper path is corrected in the artifact; the probe directory is protected against accidental overwrite.

## Next action

Extend the **test-only llama.cpp reference trace** around `llm_graph_context::build_moe_ffn` for layer 0. Capture the selected-expert gate projection, up projection, activation/clamp result, and down projection using the same token rows and selected experts as ZeroMoE's existing `L0.expert_gate`, `L0.expert_up`, `L0.expert_act`, and `L0.expert_down` probes. Compare these boundaries in order and stop at the first differing tensor. This will distinguish an input/layout issue, quantized matmul/reduction difference, activation ordering, and expert accumulation difference.

Then change one operation at a time and run same-input boundary A/B on the 3090. Do not fuse or reorder the routed expert path until these intermediate traces match or a deliberate numerical tolerance is documented and route IDs remain identical. After that, rerun teacher-forced prefill and decode through all 48 layers, including the GDN state handoff, before starting any performance tuning. Keep the user-stated 28 tok/s split and 42 tok/s hot-expert targets as later performance gates; neither was evaluated here.

Per the user's instruction, work is paused after recording this handoff. The next session should resume at the reference intermediate trace, not repeat model recovery, placement setup, VNNI ablation, or CPU-flag matching.
