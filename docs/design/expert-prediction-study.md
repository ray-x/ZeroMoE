# Predicting which experts a window will need

**Status: dead end for Qwen4Exp (decided 2026-10-27).** Router lookahead, the ridge correction and the history
predictors all fail the same test: reaching useful cold-expert recall needs many more predicted experts than the
copy or CPU budget allows (section 5). Reactive promotion (`expert-promotion.md`) already removes most misses.
Do not retry staging, pilot prefetch or ridge refits for this model unless the budget changes (for example a
much faster host-to-device path). The lane's prediction code (`ZEROMOE_QWEN_EARLY_READY_PREDICT`) stays
available for the measurement only; it is off by default.

Question: can the experts that will be computed on the CPU (cold) be predicted early enough to stage them
into VRAM or start the CPU work sooner? Measured on the palindrome and math prompts, 3090, 13,000 hot experts,
promotion on (cap 12). Traces are in `~/Projects/ZeroMoE-results/promotion-study/mtp-traces/`.

## 1. Router lookahead (existing, early-ready lane, single-token decode)

`ZEROMOE_QWEN_EARLY_READY_PREDICT=1` applies layer N+1's router to layer N's input; the engine logs
`[NEXT-LAYER-PREDICTION]` (18,800 layer-token samples each):

| prompt | recall@10 (all routed) | cold-expert recall | cold-prediction precision |
|---|---|---|---|
| palindrome | 64.9% | 44.1% (4,433 of 10,060) | 24.3% (4,433 of 18,271) |
| math | 60.9% | 44.4% (6,319 of 14,247) | 24.1% (6,319 of 26,251) |

It would cover under half of the cold work, and three of four predicted-cold experts would be wasted: staging
them would copy about 46 experts per token (about 63 MB/token), past the point where copying slows decode
(36 MB/token costs 7%; see `expert-promotion.md`).

## 2. Token-history predictors (MTP window trace, `tools/expert_lookahead_predictors.py`)

Palindrome, 463 tokens, 5.9% of routed evaluations cold. Scored on the cold evaluations of the next token:

| predictor | recall | precision | experts per layer |
|---|---|---|---|
| cold experts of the previous token | 16.3% | 17.1% | 0.6 |
| cold experts of the previous 4 tokens | 25.6% | 8.5% | 2.0 |
| cold experts of the previous 16 tokens | 31.8% | 4.1% | 7.0 |
| cold experts of the previous 64 tokens | 40.8% | 2.4% | 23.3 |
| all experts routed by the previous token | 16.3% | 36.8% | 10.0 |
| cold experts missed >= 2 times in 32 tokens | 13.2% | 5.9% | 3.6 |
| in a window: cold experts of its first token | 12.6% | 12.9% | - |

## 3. Promotion effectiveness (engine counters)

| prompt | miss | planned / confirmed | late | hits served by promoted experts | promotions never used | mean tokens to first use |
|---|---|---|---|---|---|---|
| palindrome (400 tok) | 5.35% | 3,872 / 3,863 | 0 | 13.4% | 101 (2.6%) | 47 |
| math (400 tok) | 7.56% | 5,049 / 5,037 | 0 | 39.5% | 49 (1.0%) | 36 |

## Reading

* The reactive promotion already beats every history predictor: it removes about 90% of the misses a static
  set would have, while the best predictor covers 44% of cold work with a quarter of its guesses useful.
* A promoted expert is first used about 36-47 tokens after it lands, so promotions are not wasted (1-3% never
  used) but the working set builds slowly; a larger early budget (the burst) is the useful lever.
* Prediction is not free to act on: speculative staging is copy-bound, and speculative CPU work wastes CPU
  time that is the scarce resource. It might still pay if the CPU is idle while the GPU runs attention.
* Effect on output: a prediction only chooses what to stage or precompute, never the routing, so its only
  effect on text is the CPU-versus-GPU rounding path, the same class as promotion. Measuring that needs the
  teacher-forced gate (side session); it has not been run for promotion or prediction.

## 4. Same predictor in an MTP window (apples to apples)

Router lookahead inside a verify window (lane on, `ZEROMOE_QWEN4EXP_WINDOW_LANE=1`, palindrome, 18,941 samples):
recall@10 64.8%, cold recall 45.7%, cold precision 26.0% - the same as plain decode (64.9% / 44.1% / 24.3%).
The lower numbers in section 2 are the history predictors, which see less than the router lookahead (no
activations of the token being processed).

## 5. Can accuracy be improved?

**Ridge correction** (`bench/qwen_ridge_export.py`, learned `(x - mu) R + c` added to the pilot scores). Fitted on
router activation dumps of six prompts (`~/Projects/ZeroMoE-results/promotion-study/ridge/`; 300 decode tokens
each, router taps only via `ZEROMOE_QWEN4EXP_PROBE_ROUTER_ONLY=1`):
* held-out top-3 hit rate, mean over layers: 90.7% -> 92.8% (story held out); 84.9% -> 89.5% (leetcode held
  out, fit on five prompts). Gains concentrate in a few layers (1, 5, 9, 12, 25, 42, 47: +5 to +16 points).
* In the engine on the never-seen story prompt (fit on five other prompts), cold-expert prediction got
  **worse**: cold recall 61.5% -> 57.4%, cold precision 38.3% -> 28.2%. The fit optimizes top-3 hits, not cold
  top-10 recall, so it is not the right objective for staging. Not recommended as exported.

**Predictor variants** (`tools/expert_predictor_variants.py`; offline from the dumps, static hot set, six
prompts; recall = share of cold evaluations predicted):

| predictor | cold recall | precision | experts / token / layer |
|---|---|---|---|
| baseline: pilot top-10, cold part | 64.2% | 53.1% | 2.79 |
| top-1 / 2 / 3 / 4 / 6 cold by pilot | 21 / 36 / 48 / 57 / 70% | 49 / 42 / 37 / 33 / 27% | 1-6 |
| pilot prob >= 0.05 / 0.02 | 1.1 / 10.7% | 77.6 / 88.8% | 0.03 / 0.28 |
| history: cold missed >=2x in last 32 tokens | 65.3% | 13.6% | 11.1 |
| top-3 cold by pilot UNION history | 80.2% | 14.2% | 13.1 |
| top-6 cold by pilot UNION history | 87.0% | 13.1% | 15.4 |

* Restricting to the top-m cold experts is worse per expert than filtering the overall top-10 (48% recall at 3
  experts vs 64% at 2.79). Confident predictions are accurate but rare (the router softmax over 512 experts is
  flat: almost nothing reaches 0.1). Combining the pilot with recent-miss history reaches 80-87% recall, but with
  13-15 experts per layer and token, about 600-700 per token.
* Staging by copy is out of reach: even the baseline set would copy ~134 experts per token (~185 MB), against
  the ~12 promotions (~17 MB) per token that decode tolerates. What a predictor could still serve is starting
  the CPU work on the likely cold experts early, while the GPU runs attention: the baseline would move 64% of
  the cold compute to idle CPU time at the price of about 1.3 wasted expert evaluations per layer and token.
  That has not been built or measured.
