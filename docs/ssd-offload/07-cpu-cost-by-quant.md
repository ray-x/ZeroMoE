# CPU cost of one routed-MoE layer per quant type (2026-10-09)

`bench/expert_cost_by_quant_bench.cpp` (+ `q2_0` = ggml type 42 added to its list), accepted ggml (`moestream-build-upstream-master/ggml-build`,
AVX-512 VNNI), Xeon W-2235, 8 threads, decode shape (1 token, 10 routed slots), D=2560, F=640 (768 for 256-block types).

| type | MB/expert | us per layer, 10 experts | us/expert | effective GB/s |
|---|---|---|---|---|
| q4_0 | 2.76 | 737 | 74 | 37.5 |
| q3_k | 2.53 | 774 | 77 | 32.7 |
| q4_k | 3.32 | 794 | 79 | 41.8 |
| iq4_nl | 2.76 | 788 | 79 | 35.1 |
| iq4_xs | 3.13 | 847 | 85 | 37.0 |
| **q2_0** | 1.38 | 937 | 94 | 14.8 |
| q5_k | 4.06 | 973 | 97 | 41.7 |
| q6_k | 4.84 | 1065 | 106 | 45.4 |
| q8_0 | 5.22 | 1091 | 109 | 47.9 |
| iq1_m | 1.29 | 1140 | 114 | 11.3 |
| **iq2_xxs** | 1.52 | 1288 | 129 | 11.8 |
| iq2_xs | 1.70 | 1313 | 131 | 13.0 |
| **iq3_xxs** | 2.26 | 1660 | 166 | 13.6 |

The 4-8 bit types stream at 33-48 GB/s (memory-bound). The 1-3 bit codebook types (IQ1/IQ2/IQ3) are compute-bound at 11-15 GB/s: IQ3_XXS
costs 1.8x Q2_0 and 2.2x Q4_0 per expert although it moves less data than Q4_0; IQ2_XXS (the GLM-5.3-Flash AJ file) costs 1.4x Q2_0.
Per expert, the cheapest CPU types are Q4_0 / Q3_K / Q4_K / IQ4_NL (74-79 us) despite 1.7-2.2x the bytes of IQ2_XXS.

## GLM-5.3-Flash expert shape (2026-10-13): the Qwen ranking does not transfer

Same bench with `BENCH_D=4096 BENCH_F=2048 BENCH_NE=64 BENCH_TOPK=8`, 8 threads, ggml with the IQ2_XXS AVX-512VL dot (real F is 2304,
so absolute numbers are ~12% low; the ranking holds). At 6-17 MB per expert the 4-8 bit types hit the DDR streaming limit (~55-60 GB/s):

| type | MB/expert | us/expert | GB/s | note |
|---|---|---|---|---|
| iq1_s / iq1_m | 4.9 / 5.5 | 292 / 300 | 17-18 | compute-bound |
| **q2_k** | 8.3 | **153** | 54 | fastest, but see noise below |
| iq2_xxs (stored) | 6.5 | 250 | 26 | compute-bound, ~2.2x above its 114 us memory floor |
| iq2_s | 8.1 | 328 | 25 | |
| q3_k | 10.8 | 217 | 50 | |
| q4_0 / iq4_nl / iq4_xs / q4_k | 13-14 | 235-265 | 53-60 | no faster than iq2_xxs |
| q5_k | 17.3 | 286 | 60 | |
| iq3_xxs / iq3_s | 9.6 / 10.8 | 424 / 528 | 21-23 | |

Re-quantisation noise of the stored IQ2_XXS experts (`bench/requant_noise_probe.cpp`, layer 10 expert 5, RMS error relative to the
stored weights): q2_k 0.295, q3_k 0.160, q4_0 0.124, q4_k 0.034.

Consequence: a hybrid "Q4_0 on the CPU" tier buys nothing (same speed, 12% noise); Q2_K is 39% cheaper but adds ~30% noise on top of
the IQ2 error; Q4_K is near lossless but not faster. Without an original higher-precision source there is no quant swap that is both
faster and faithful. The remaining CPU headroom is a faster IQ2_XXS kernel (250 us vs a 114 us memory floor).
