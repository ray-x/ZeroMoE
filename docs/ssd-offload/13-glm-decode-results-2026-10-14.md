# GLM-5.3-Flash decode performance: results record (2026-10-14)

Box: Xeon W-2235 (6C/12T, Cascade Lake, AVX-512 VNNI), 125 GB DDR4, RTX 3090 (PCIe Gen3 x16, ~12.2 GB/s host->device) + RTX 5060 Ti (PCIe Gen3 x8, ~6.9 GB/s); the W-2235 platform is PCIe 3.0 only, so the 3090 never runs Gen4.
Model: GLM-5.3-Flash AJ-IQ2_XXS (87 GB), experts offloaded (SSD -> DDR pool -> VRAM hot sets), full-graph decode.
Branch `graph-framework` (master fast-forwarded to `1842a5c`).

## Method (how the numbers were taken)
- tok/s = 1000 / mean wall-clock ms per token over the **last 30 decoded tokens**, from consecutive `FULL-DECODE-TIME` log lines.
  Averages over all 40-48 tokens include slow warm-up tokens and read ~0.3-0.5 tok/s lower.
- Compared configurations always decode **identical tokens** (teacher forcing: `MOESTREAM_TEACHER_FORCE_TOKEN_FILE`;
  `bench/data_glm_tf48.tokens`, or tokens from a free run of the same prompt via `bench/ranking_ab_prompt.sh`).
  Free-running text alone moves the same configuration between 13.8 and 16.0 tok/s.
- Held-out = prompt not in the ranking file used (train-only file `usage_train30`); benchmark sequence = "Explain how a
  mixture-of-experts transformer routes tokens ...", 48 tokens.

## Results
Default binary (`build_gk`: in-tree ggml with the AVX-512VL IQ2_XXS kernel + patch 0006), no library overrides, recipe env
(`bench/glm_recipe_env.sh`: hot set 15.0 GiB on the 5060 Ti, 10 CPU threads, auto pool ~76 GiB, second assist ring on the 5060 Ti):

| input | before | after |
|---|---|---|
| benchmark sequence | 14.85 tok/s | **15.31** (15.43 with the kernel library) |
| held-out 094 dockerfile | 12.32 | **15.03** |
| held-out 046 css | 11.69 | 13.19 |

Peak (best windows found in the logs, recipe): best 30-token stretch 17.1 tok/s, best 10-token stretch 19.0, fastest single token
~24 tok/s (42 ms). Typical sustained: 13-15 tok/s depending on how well the reply matches the resident set (per-token time
42-99 ms with the number of routed experts outside VRAM).

Qwen3.8-Flash-Next (Q2_0, 3090 only, 96 tokens, tokens identical): baseline 29.81 / 30.66 vs new 30.94 / 30.81 tok/s (+2%, within
noise); GPU-trunk bound, unaffected by the GLM changes. Identity gate on the final binary: Qwen logits max KL 0.0, free-running
tokens identical; GLM `lru` and `lane` logits and tokens identical vs the frozen baseline.

## Single 3090 (no 5060 Ti), from existing logs
Not re-measured with the final recipe: these are older-config runs (lane + GPU assist on, 58 GiB pool, 8 CPU threads, previous ranking,
mixed ggml builds), taken from `logs/perf/sweepA_A_1`, `sweepB_*`, `assist_tf`, `assist3_*`; steady = last 30 tokens.

| window | single 3090 | dual GPU (recipe) |
|---|---|---|
| sustained (last 30 tokens) | 12.2-13.3 tok/s | 15.0-15.4 (benchmark-like text) |
| best 30-token stretch | 13.3 | 17.1 |
| best 10-token stretch | 14.8 | 19.0 |
| fastest single token | 52 ms = ~19 tok/s | ~42 ms = ~24 tok/s |
| 3090-only without assist (older runs) | 9.9-10.6 | - |

So a single 3090 exceeds 15 tok/s only over short bursts (single tokens, ~10 tokens at best 14.8) and averages ~12-13 over a reply.
Reason: the 3090 alone keeps ~14.7 GiB of hot experts (vs 28.9 GiB with the 5060 Ti), so more routed experts go through the CPU lane
and assist. The final recipe (refit ranking, kernel in the default binary, 10 threads) on a single 3090 has not been measured; an
estimate of +0.5 to +1 tok/s over these numbers is a guess.

## What produced the gain (this session, GLM, dual GPU)
| change | effect |
|---|---|
| split load, IQ2_XXS kernel, bigger DDR pool, graph prefill | earlier work: 4.3 -> ~10 tok/s on the 3090 alone |
| 5060 Ti hot set 14.2 GiB, lane on its layers, GPU assist (3090 ring) | 3090-only 10.1 -> 14.7 tok/s |
| assist only for IQ/Q2_0 expert layers | GSQ-RCO file 88.2 -> 70 ms/token (K-quants stay on the CPU) |
| hot set 15.0 GiB, second assist ring (5060 Ti), threads 10 | +0.3 tok/s (66.9-67.3 ms vs 68.3) |
| ranking refit on 100 prompts (decode-only routing) | held-out +1.4 / +2.5 tok/s; neutral on the benchmark prompt |
| engine linked a ggml without the AVX-512 kernel | +0.5 tok/s (14.91 -> 15.43) |
| GSQ-RCO 3.0-bit file (Q2_K/Q3_K/Q4_K, Q8_0 dense) | parity with AJ: 68.9 ms with assist 1 (default now: assist off for K-quants, 70.1 ms) |

## Tried and not worth it (details in doc 11/12)
Dense on the 5060 Ti (-8 ms), Q4_0/Q2_K cold tier (no faster at GLM shape / 30% noise), THP, dynamic resident set (-1% cold routes at
8 promotions/token), assist slots > 3, cost-model assist split, IQ2_XXS micro-restructurings (+8% microbench, noise end to end),
prompt-lookup speculation (1.03 words/pass), MTP (no head in either checkpoint).

## Open
- The user's shared build `~/Projects/moestream-build-upstream-master` still links the Oct 4 ggml (no kernel): ~14.9 tok/s until
  rebuilt from `llama.cpp-moestream-integration` (`7bd457fa1`).
- Prompts far from the surveyed corpus stay at 13-14 tok/s; a usage ranking from the real workload is the remaining lever
  (`bench/make_survey_batches.py`, `bench/glm_route_survey.sh`, `bench/ranking_holdout_eval.py --write`).

## 5060 Ti efficiency round (2026-10-14; code kept on branch `assist-prestage-experiment`, nothing merged)
Where the 5060 Ti's time goes: ~3 ms/token of kernels out of ~65 (4-5% busy), its link carries ~1 assist expert per layer at 6.9 GB/s.
Its +20% over a single 3090 is residency (28.9 vs 14.7 GiB of hot experts), not compute. Same fixed sequence, last 30 tokens, recipe env:

| experiment | result |
|---|---|
| layer split, 5060 Ti holds only layers >= 24 / 31 / 36 (3090 the first layers) | 65.3 / 64.5 / 64.4 ms vs 65.4 (15.31 / 15.51 / 15.52 vs 15.30 tok/s); trunk wait -1.2 / -2.5 / -6.2 ms, lane compute +2.5 / +3.9 / +4.5 ms |
| non-contiguous layer sets from an offline optimiser (15 and 19 layers incl. 3-12) | 65.1 / 64.7 ms vs 64.6 / 64.9: neutral; the model predicted 1.4-4.7 ms better, the benchmark prompt's lane cost rose by as much as the hops saved |
| assist prestaging: pilot predicts next layer's cold experts, copied to the ring during the trunk | lane compute 38.1 -> 30.3 ms, 68% of assisted experts already in the ring, but wall 64.7 -> 95 ms (an unexplained ~0.6 ms/layer per staged expert outside the lane's wait/compute); default off |
| first prestage build | 381 ms: the extra predictor tensors overflowed the exactly sized dense arena and the LM head silently fell back to the CPU (fixed in the experiment branch) |

Kernels: with the 5060 Ti busy ~3 ms/token, kernel tuning there cannot move the total; the native-arch (sm_120) build was not tried.
Pin layout: assist copies run at the PCIe limits (3090 ~12 GB/s per 2 MB plane copy, 5060 Ti 6.9 GB/s), planes are separate regions,
three copies per expert; no layout change was found that helps. Communication: cross-GPU hop removal is real (-6 ms of trunk wait) but is
paid back by lane compute because the 5060 Ti's set then matches the hot experts worse.

### Prestage: root cause of the stall, and why it still does not pay (same day, branch `assist-prestage-experiment`)
- **Stall cause (found):** layers that have a second-GPU branch do cross-GPU peer copies (hidden to the 5060 Ti and back) inside their
  trunk. The staged expert DMA (1-1.6 ms per layer) occupies the same copy engines/PCIe path, so those small copies queue behind it:
  between-lane host time went 10.2 -> 38.0 ms/token (+0.67 ms per layer). Not the LM head after the arena fix, not `cudaFree` (those
  are the legacy prefill), not `CUDA_DEVICE_MAX_CONNECTIONS`, and not the order of the lane's output upload.
- **Mitigation:** prestage only layers without a second-GPU branch (layer-split layout): gap back to 8.1 ms, 70.1 ms vs 64.5 ms base.
- **Why no gain:** with 61% of assisted experts pre-staged, lane compute did not fall (42.7 vs 41.4 ms): the in-window copies are not
  the straggler, the CPU graph is. Letting staged hits assist on top of the normal share moves time back to the assist join
  (compute 46-49 ms, 74 ms/token). The lane is on a balanced plateau; prestage cannot shift it.
- Net: no measured improvement in 5060 Ti efficiency from layout, prestage or kernels; the experiments and tools stay on the branch.

## Layer split adopted: the 5060 Ti holds layers >= 31 (2026-10-14)
Order-balanced repeats on identical tokens, last 30 tokens, tok/s, binary with the AVX-512 kernel, recipe env (hot set 15.0 GiB, 10
threads), ranking built without the prompt:

| input | current layout | 5060 Ti holds layers >= 31 | 5060 Ti holds layers >= 36 |
|---|---|---|---|
| benchmark sequence (n=4 / 2 / 4) | 15.37 (15.25-15.42) | 15.40 (15.38-15.43) | 15.40 (15.37-15.48) |
| held-out css 046 (n=3 each) | 14.71 (14.70-14.71) | **15.20** (15.19-15.22), +3.4% | 14.84 |
| held-out dockerfile 094 (n=3 each) | 15.31 (15.29-15.33) | **15.76** (15.67-15.83), +3.0% | 15.26 |

Mechanism: a layer with hot experts on both GPUs pays cross-GPU copies of the hidden state inside its trunk; with the cut, layers 3-30
are 3090-only (no hop; trunk wait 21.5 -> 19.0 ms/token) and layers 31-44 are 5060 Ti-only, at the price of a slightly worse hot-set match
(lane compute 37.7 -> 41.7 ms). The trade is neutral for in-domain text and positive for out-of-domain text, whose routes are less
concentrated. Shipped: `MOESTREAM_GLM5NEXT_GPU1_MIN_LAYER=31` (default in `bench/glm_recipe_env.sh`), the 3090 ranking with layers >= 31
zeroed (`glm53flash_usage_base_plus_decode100_3090_lt31.txt`) and `pin_global_floor_per_layer = 0` in the shipped config.

### Pin layout / PCIe links (2026-10-14, `nvidia-smi` + a standalone copy microbenchmark)
Both links negotiate PCIe **Gen3** (`pcie.link.gen.max = 3`; 3090 x16, 5060 Ti x8); the Xeon W-2235 has no Gen4. Host->device copies of
2.16 MB expert planes from 2 GiB of buffers: 12.2 GB/s on the 3090 and 6.9 GB/s on the 5060 Ti for every host layout tried
(`cudaMallocHost`, anonymous 4 KiB pages + `cudaHostRegister`, 2 MiB THP + register, randomly scattered 4 KiB pages + register),
i.e. the copy rate is the link limit and the arena's page layout / pinning method cannot improve it. Earlier notes that call the 3090
"PCIe4" are wrong; the assist copies were already running at the Gen3 x16 limit.
