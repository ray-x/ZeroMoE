# Promoting experts into the Qwen4Exp graph arena (design study)

Qwen4Exp's full-graph decode keeps a static hot set in VRAM (Option B: global top-K from a usage
file recorded on other prompts); everything else is computed on the CPU from host RAM. The legacy
tiered VRAM cache (`GpuTopology`, eviction score `last_used_clock + 64 * hit_count`) is disabled on
this path. This study asks how much a promotion mechanism could remove CPU expert work, and which
design to build: **A** the best design regardless of origin (batched swaps from decayed counts, as in
Strata's `adapt()`) or **B** an extension of ZeroMoE's own tiered cache (reactive, hit-count
threshold, eviction score).

## Method

Routing traces from plain S=1 decode of six prompts (code review, palindrome, LeetCode, story, math,
B-tree; 245-499 tokens each), 13,000 hot experts, 3090, Q2_0 hybrid recipe. Per token and layer the
10 routed ids and whether each was in the static VRAM set were recorded (a temporary hook in the
router-usage tap, not committed). `tools/expert_promotion_sim.py` replays them with the real slot
count per layer and counts the routed experts that would still run on the CPU.

## Findings

* The static hot set misses **24%** of routed experts across these prompts: 8% for code review but
  42-51% for math and story. Plain decode follows: 33-37 tok/s for the 11-15% prompts, 24-25 tok/s
  for math and story (router-usage tap on).
* An oracle static set profiled on the same prompt would miss about 1%, so the working set of one
  reply is small.
* **Eviction tuning does not matter at this scale.** Frequency weight (0..4096), decay interval
  (512..32768) and the recency/frequency blend give identical results: each layer holds hundreds of
  never-used slots that are evicted first. The levers are the promotion trigger and the per-token
  budget.
* Same budget (about 13-17 MB moved per token), CPU expert evaluations removed:

  | policy | per-prompt | mixed 2,703-token session |
  |---|---|---|
  | A: batch every 8 tokens, 64 swaps | -72% | -66% |
  | B: promote at 2 misses, 6-8 per token, rotating layer order | -68 .. -73% | -61 .. -66% |
  | B with a layer-ordered cap | -54 .. -66% | -44 .. -53% |

  The layer-ordered cap lets early layers consume the budget; rotating the start layer (or one
  promotion per layer per token) closes the gap to A.

## Eviction score under real evictions (MTP-style steps)

The first sweeps never forced a real eviction (each layer has hundreds of never-used slots), so every
score looked identical. `tools/expert_promotion_eviction_sim.py` removes the slack: the resident set
holds only experts that some prompt uses, a step is 3 tokens (MTP commits ~3 tokens), budget 6 promotions
per token plus the start-of-reply burst. Four random orders of ten prompts (repeating topics), CPU expert
evaluations removed against the static set:

| eviction score | mean over 4 orders |
|---|---|
| pure LRU | 61.3% |
| ZeroMoE tiered-cache constants (`last_used + 64*hits`, halve every 4,096 hits) | 61.3% |
| `last_used + 4096*hits`, no practical decay | **64.1%** |
| `last_used + 16384*hits`, no practical decay | 63.6% |
| `last_used + 65536*hits` | 57.0% (frequency too sticky) |
| true LRFU, half-life 250k accesses (~520 tokens) | 64.3% |

With the old constants the blend is pure recency: its hit counts halve every 4,096 hits, i.e. every
~9 tokens at ~480 routed experts per token. A frequency weight of about 4,096 (one hit is worth ~8.5
tokens of recency) with no practical decay matches true LRFU, so our own score stays and only the
constants change (`PromotionConfig`). Step size matters little (1 token 64%, 3 tokens 63%, 8 tokens 61%).

Recorded traces, prompts and the simulators live in `~/Projects/ZeroMoE-results/promotion-study/`;
the six prompts are also in `bench/prompts/mtp_suite/`.

## Decision

The gap between A and B is within a point of miss rate, so build **B**: promote a missed expert
after 2 misses, at most 6-8 promotions per token spread fairly across layers, victim by the existing
eviction score, with a switch that freezes the hot set for validation.

## Measured costs (3090, PCIe 3.0 x16, Xeon W-2235)

* The host expert arena is pinned: Qwen4Exp registers it with `cudaHostRegister` by default (startup log
  `[GRAPH-RAM] layer N arena: ... pinned`); the code records pageable 4.9 GB/s vs pinned 10.6 GB/s.
* One expert is gate+up+down = 3 x 460,800 B = **1.38 MB** (24,576 experts = 34 GB, matching Strata's Q2_0 figure).
* Copy interference (`ZEROMOE_PROMO_PROBE_N`, story prompt = 51% static misses, plain S=1 decode, 500 tokens,
  N experts copied host->device on a side stream after every token, no late waits in any run):

  | promotions/token | MB/token | decode tok/s | vs baseline |
  |---|---|---|---|
  | 0 | 0 | 26.47, 26.75 | - |
  | 6 | 8.3 | 26.32 | -0.6% (noise) |
  | 26 | 36 | 24.65 | -7% |
  | 60 | 83 | 22.00 | -17% |

  A steady budget of 6 per token is free; a start-of-reply burst of ~26 costs ~7% only while it lasts;
  interference comes from sharing host RAM bandwidth with the CPU experts.

## Not covered

The simulation assumes a promoted expert is resident immediately and ignores the cost of the copies
(host RAM bandwidth is shared with the CPU experts) and the numerics of moving an expert between CPU
and GPU (Q8_K vs Q8_1 activation rounding), so output depends on cache history. 
