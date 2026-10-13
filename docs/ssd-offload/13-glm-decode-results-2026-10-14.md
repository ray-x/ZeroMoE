# GLM-5.3-Flash decode performance: results record (2026-10-14)

Box: Xeon W-2235 (6C/12T, Cascade Lake, AVX-512 VNNI), 125 GB DDR4, RTX 3090 (PCIe4 x16) + RTX 5060 Ti (PCIe3 x8).
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
