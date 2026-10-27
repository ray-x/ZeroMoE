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

## Engine results (swap-b `e87076e` + env overrides; 3090 only, 13,000 hot experts, Q2_0)

Source: `docs/design/expert-promotion.md` tools; raw logs, configs and run scripts in
`~/Projects/ZeroMoE-results/promotion-study/engine-matrix/`. Binary: `ZeroMoE-wt-swap-b/build/zeromoe`
(Release, sm_86, patched ggml 8baabd53b from `build_recovery_ggml`). Each cell is one run of 246-500 tokens,
greedy, taken with the machine idle (a wrapper waits for load < 2.5 and MemAvailable >= 108 GB); repeated
identical runs differ by about +-3 tok/s, so differences below ~10% are noise. The first pass of this matrix was
taken while another session was reading the 62 GB model and showed fake regressions (-42%); those numbers were
discarded.

Wall decode rate (tok/s, includes one-time decode-graph/draft setup), promotion `off` -> `on`:

| prompt | static miss | plain | MTP (3 drafts) | MTP accept | MTP on / plain off |
|---|---|---|---|---|---|
| story | 51% | 24.2 -> 42.1 | 18.9 -> 35.7 | 45% | 1.48x |
| math | 42% | 27.4 -> 43.9 | 32.6 -> 52.8 | 83% | 1.93x |
| btree | 11% | 42.2 -> 47.6 | 43.3 -> 51.7 | 64-66% | 1.22x |
| leetcode | 15% | 42.3 -> 40.0 | 47.5 -> 49.1 | 87-89% | 1.16x |
| code review | 8% | 44.6 -> 48.5 | 48.2 -> 52.8 | 68-72% | 1.18x |
| palindrome | 10% | 42.7 -> 48.6 | 52.8 -> 56.2 | 81% | 1.32x |

* MTP gains from promotion: +89% story, +62% math, +19% btree, +10% review, +6% palindrome, +3% leetcode
  (the last two are inside the noise). The more the static set misses, the more promotion helps; where the
  static set already fits (code prompts) it is roughly neutral to +10%.
* Promotion removes 70-90% of the misses within a reply (e.g. story 51% -> 10%, math 42% -> 12%).
* With promotion on, MTP beats plain on math (+20%), btree (+9%), review (+9%), palindrome (+16%), leetcode
  (+12-23%), but not on story: acceptance is only 45% there, so a 4-token window costs more than it returns
  (35.7 vs 42.1).
* Eviction constants (old 64 / halve every 4,096 hits vs new 4096 / no decay) give identical miss rates and
  acceptance within one request (story 9.84%, math 11.84%) and only noise-level speed differences. They can
  only matter once the resident set carries over between requests or topics, which this implementation does not
  do yet (the decode graph, and so the policy, is rebuilt per request).

## Refresh matrix with repeats (binary `b5eee36`, fused cold op on)

`tools/bench_qwen4exp_mtp_suite.py`: six prompts, plain and MTP (3 drafts), fresh process each, exactly 256 generated
tokens, 3 repeats, medians of decode-wall tok/s (includes ~240 ms of draft setup for MTP). RTX 3090 only, Q2_0,
K=13,000, context 1536. Two configs: static hot set, and `qwen4exp_expert_promotion = "on"`. Raw logs, configs and
manifests: `~/Projects/ZeroMoE-results/promotion-study/matrix-6751f3c/{static,promo}/` (binary SHA
`af6bf5c7f11a...`, source `b5eee36`, ggml `8baabd53b`).

| prompt | plain | plain + promo | MTP | MTP + promo | MTP+promo vs plain+promo | MTP accept | tok/cycle |
|---|---|---|---|---|---|---|---|
| palindrome | 48.0 | 46.0 | 60.5 | **62.5** | 1.36x | 79.6% | 3.37 |
| code review | 49.5 | 48.0 | 55.4 | **59.8** | 1.25x | 68.7% | 3.05 |
| btree | 45.8 | 49.3 | 50.6 | **60.0** | 1.22x | 69.4% | 3.08 |
| leetcode | 43.2 | 44.3 | 52.7 | **57.2** | 1.29x | 84.3% | 3.51 |
| math | 30.8 | 44.2 | 33.0 | **55.1** | 1.25x | 77.7% | 3.32 |
| story | 25.6 | 44.8 | 18.9 | **42.9** | 0.96x | 50.3% | 2.51 |

* Promotion helps MTP most where the static set misses most: +127% story, +67% math, +19% btree, +9% leetcode,
  +8% code review, +3% palindrome (palindrome and leetcode are within noise).
* MTP over plain, both with promotion: 1.22-1.36x on five prompts; story is 0.96x (acceptance 50%, 2.5 tokens
  per window).
* These 256-token medians are 5-8% below the best single runs of 463-500 tokens quoted earlier (67-68 tok/s on
  palindrome and code review): short runs carry the fixed ~240 ms setup and the promotion warm-up, and single
  runs pick up the +-4 tok/s noise. 60-62 tok/s is the honest figure for a 256-token reply on easy prompts.
* Strata, measured by the user on this machine: ~68-70 tok/s (31 and 512 token decodes).

## Cross-request promotion (studied, not built)

The decode graph, and with it the promotion policy, is rebuilt for every request, so each request starts again from
the static hot set (two requests in one process: 47.1 then 47.5 tok/s, no carry-over). Offline from the recorded
traces (first 256 tokens, same policy and constants), the share of routed experts that run on the CPU:

| prompt | cold start | start from the set learned on the same topic | start from another topic | experts to restore |
|---|---|---|---|---|
| palindrome | 4.4% | 1.2% | 4.7% | 1,911 (2.6 GB) |
| code review | 3.1% | 0.8% | 7.1% | 1,393 (1.9 GB) |
| math | 7.0% | 1.0% | 7.3% | 3,221 (4.4 GB) |
| btree | 3.6% | 0.8% | 5.8% | 1,699 (2.3 GB) |
| leetcode | 6.7% | 1.8% | 18.2% | 2,614 (3.6 GB) |
| story | 6.6% | 0.6% | 7.4% | 2,648 (3.7 GB) |

* Same topic: 3,900-7,300 fewer cold evaluations per 256 tokens, roughly 0.4-0.7 s at about 0.1 ms each.
* The set has to be copied back before the first token (2-4 GB, about 0.2-0.4 s at 10 GB/s, not hidden behind
  decode), so the net gain is a few percent, and only when the topic repeats.
* A different topic is worse than a cold start (leetcode 6.7% -> 18.2%): the carried-over set evicts the static
  experts the new prompt needs.
* Decision: not built. The warm-up is already hidden behind decode; persistence would trade it for an exposed copy.

Multi-request check (plain decode, two identical prompts in one process, promotion on vs off, 200 tokens):
K=13,000 and K=11,000 give identical text for both requests; K=12,500 matches until character 785 of ~810 and then
differs, as static mode already does between its own first and second request (809 vs 812 characters). I found no
sign that a later request rebinds promoted bytes. The binding code still treats the arena as immutable and keys
reuse on the source pointers, so this is a tested behaviour, not a guarantee; a request that does not release the
arena at prefill would be the case to watch.

## Output check under promotion (teacher-forced, `tools/check_adaptive_placement_accuracy.py`)

Static hot set (K=13,000) generates 256 tokens per prompt; the same tokens are forced through the same binary with
`qwen4exp_expert_promotion=on` (cap 12, burst 20) and the per-step logits are compared. Binary `6751f3c`, 3090,
six prompts of `bench/prompts/mtp_suite`; raw per-token data in
`~/Projects/ZeroMoE-results/promotion-study/gate-promotion/`.

| prompt | top-1 agreement | logit cosine | mean KL |
|---|---|---|---|
| btree | 99.61% | 0.999499 | 0.00065 |
| leetcode | 100.00% | 0.999122 | 0.00024 |
| math | 98.44% | 0.999167 | 0.00026 |
| palindrome | 100.00% | 0.999376 | 0.00040 |
| code review | 99.22% | 0.998943 | 0.00067 |
| story | 98.44% | 0.999520 | 0.00110 |
| **macro** | **99.28%** | **0.999271** | **0.00055** |

For scale, the side session measured static K=13,000 vs K=11,000 at 99.09% top-1, cosine 0.999392, KL 0.00059:
promotion moves experts between CPU and GPU (different activation rounding) by about as much as changing the hot-set
size, and so it can change the sampled text on near-ties. It does not change which experts are routed.
MTP output stays exact against the same placement (verified with promotion off).

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
