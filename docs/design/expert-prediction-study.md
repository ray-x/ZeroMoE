# Predicting which experts a window will need

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
