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

## Decision

The gap between A and B is within a point of miss rate, so build **B**: promote a missed expert
after 2 misses, at most 6-8 promotions per token spread fairly across layers, victim by the existing
eviction score, with a switch that freezes the hot set for validation.

## Not covered

The simulation assumes a promoted expert is resident immediately and ignores the cost of the copies
(host RAM bandwidth is shared with the CPU experts) and the numerics of moving an expert between CPU
and GPU (Q8_K vs Q8_1 activation rounding), so output depends on cache history. Whether the host
arena is pinned for full-speed async copies is unchecked.
