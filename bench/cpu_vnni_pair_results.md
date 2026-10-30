# CPU VNNI MoE pair experiments

> Historical reproduction note (2026-10-30): `ZEROMOE_Q2_0_DISABLE_VNNI` was removed. The old forced-fallback command below documents the original comparison and is no longer a supported runtime invocation.

This note records the CPU dot-kernel experiments that informed the current
scope. The only VNNI MoE signature retained for runtime use is
`Q2_0 / Q2_0 / Q2_0`. The IQ4_NL and signed-Q8 VNNI candidates and the
experimental hybrid Q8_0 fused paths were removed after testing; unmeasured
signatures remain on the existing GGML path.

## Test setup

- Host: Intel Xeon W-2235 (Cascade Lake), AVX-512 VNNI available.
- Release build, `-O3`, GGML built for the local CPU (`GGML_NATIVE=ON`). The
  custom VNNI functions use function-level ISA targeting and runtime CPU
  detection; the whole binary does not need a global VNNI compiler flag.
- Focused Q2 workload: genuine GGML-quantized Q2_0 weights, `D=2560`,
  `FF=640`, 10 experts, five OpenMP workers. Dot-only timing includes 38,400
  rows per pass and excludes quantization, SwiGLU, routing, and output reduce.
  Timings were alternated to reduce order bias.
- Other dot tests used the same dimensions and worker count, with 25,600
  down rows/pass for the IQ4_NL and Q8_0 candidates.
- Results are host/build/workload specific; a model smoke test remains the
  final check before enabling a runtime path for a deployment.

## Results

| Signature / candidate | GGML | VNNI or fused candidate | Outcome |
| --- | ---: | ---: | --- |
| `Q2_0 / Q2_0 / Q2_0`, dot-only | 2.422–2.437 ms/pass | 1.578 ms/pass | Latest run: VNNI 1.54x faster; correctness passed. Earlier paired median was ~1.52x. |
| `Q2_0 / Q2_0 / Q2_0`, fused direct route | 2.572 ms/call fallback | 1.682 ms/call VNNI | Latest run: VNNI 1.53x faster; fused route and numerical checks passed. |
| IQ4_NL down-dot, IQ4_NL × Q8_0 | paired runs: 0.332–0.470 ms/pass | 0.457–0.641 ms/pass | VNNI ~1.38x slower by paired median ratio; correctness passed. Candidate removed. |
| Q8_0 down-dot, Q8_0 × Q8_0 | 0.343–0.380 ms/pass | 0.528–0.652 ms/pass | VNNI ~1.5–1.8x slower; correctness passed. Candidate removed. |
| IQ2_XXS VNNI prototype | AVX2 GGML-port 0.1955 us/call; FMA 0.3246 us/call | 0.7553 us/call | VNNI lost, and this was not production-equivalent: prototype used Q8_0 activations while production IQ2_XXS uses Q8_K. Not integrated. |
| `IQ2_XXS / IQ2_XXS / Q8_0`, fused GGML experiment | separate teams: 0.998–1.211 ms/pass | one-team fused: 1.012–1.197 ms/pass | No consistent advantage; exact output match. Experimental path removed. |
| `IQ4_NL / IQ4_NL / Q8_0`, fused GGML experiment | separate teams: 0.902–0.967 ms/pass | one-team fused: 0.871–0.899 ms/pass | Small apparent gain in these runs, not enough evidence for a runtime path; exact output match. Experimental path removed. |

There is no result for `Q2_0 / Q2_0 / IQ2_XXS`,
`Q2_0 / Q2_0 / IQ4_NL`, or full IQ2_XXS/IQ4_NL all-projection VNNI paths.
These combinations are not implemented as VNNI paths and should continue to
fall back to GGML.

## Why the losing candidates were slow

The Q8_0 candidate benchmarked here did **not** use the `+128` weight-offset
scheme. It used `abs(weight)` and sign-conditioned activation negation to
express signed × signed math through `vpdpbusd`. That needs extra sign/magnitude
work and wider-vector assembly around the dot. Q8_0 also has an independent
scale per 32 values, so the implementation still had to apply block scales
through floating-point accumulation; it could not defer all scaling to one
integer reduction at row end. GGML's native Q8_0 dot already uses efficient
packed multiply-add instructions on this host. The direct-dot benchmark put
the same outer OpenMP team around both kernels, so parallel-region creation
was not the cause of the measured VNNI loss.

IQ4_NL has similar signed-codebook conversion work, plus nibble lookup and
cross-lane unpacking before VNNI. Those costs outweighed the dot-product
throughput. Q2_0 was a better fit: unpacked codes are already unsigned 0..3,
and the kernel can correct its zero-point while keeping partials in vector
lanes and reducing once per row. This is why Q2_0 is retained while the other
formats are not.

## Re-running the retained comparison

The isolated `ZEROMOE_TEST_ONLY_Q2_0_VNNI` selector is retired. The native
harness now runs the Q2_0 dot, fused-route, and numerical checks in its normal
suite:

```bash
cmake --build build-cuda --target zeromoe_tests
./build-cuda/zeromoe_tests
```

VNNI is selected automatically when supported; other CPUs use the linked GGML
dot. The historical forced-fallback invocation used
`ZEROMOE_Q2_0_DISABLE_VNNI=1`; that runtime A/B override is also retired.


## Worktree smoke timing (2026-10-30)

On the same Xeon W-2235, the focused `ZEROMOE_TEST_ONLY_Q2_0_VNNI=1 build-cuda/zeromoe_tests` run reported exact output agreement and selected VNNI. Its built-in single-pass timing reported GGML 0.620-0.713 ms/pass and VNNI 1.711-1.789 ms/pass (38,400 rows, five threads). This is not a controlled replacement for the paired experiment above: it uses the current dirty ZeroMoE snapshot and pinned GGML build, and the harness timing is an unpaired smoke measurement. It conflicts with the older 1.54x VNNI win; do not treat either ratio as transferable without a matched repeat.
