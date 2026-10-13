# GLM-5.3-Flash: scope for multi-token (speculative) decode (2026-10-14)

Status: proposal, nothing built. Why: with a static resident set, far-domain text sits at 13-14 tok/s because ~35% of routed experts
(~117 per token) are evaluated on the CPU lane/assist rings, which are already on a balanced plateau (see doc 11). The only lever that
amortises expert work across tokens is verifying several tokens per pass.

## What the checkpoints allow
- No MTP/NextN head in either file (AJ declares `nextn_predict_layers = 0`; GSQ-RCO has no such key; no `blk.45`). MTP is out.
- No small draft model for GLM-5.3-Flash is on disk. Options are model-free drafts: prompt-lookup / n-gram (good on code and
  repetitive text, poor on prose) or a small separate model (needs tokenizer-compatible weights, not available here).

## Expected gain (estimate, not measured)
- Expert work per verification pass of k tokens is the union of their routed experts. Measured on the trace: 8 routes/layer/token;
  unions over 2-3 consecutive tokens are ~1.5-2x (not 2-3x) of one token because adjacent tokens share experts.
- Trunk (21 ms/token wait) and per-token fixed costs are paid once per pass. With acceptance a (fraction of drafted tokens kept),
  throughput ~ (1 + a*(k-1)) tokens per pass at ~(1.3-1.6)x the single-token pass time. For k=3, a=0.5 (code-like text):
  2.0 tokens / 1.5 ~ 1.3x. For prose with prompt-lookup, a is near 0 and there is no gain (a small loss).
- So: ~1.3x on code/boilerplate (css/html would benefit), ~1.0x on prose.

## Work required
1. Graph: a decode graph for S=k (k<=4) with KDA recurrent-state handling for k steps (state must be checkpointed per step so a rejected
   draft can be rolled back) and per-token KV/DSA indexer positions. The prefill graph has S>1 but is position-zero only today.
2. Lane: the CPU lane, split-load and GPU assist are written for S=1 (k routes per layer). They need the union-of-routes form.
3. Verification/sampling: accept-longest-prefix against the target's greedy (or sampled) tokens; draft source (n-gram table over the
   prompt+generated text).
4. Gate: identity vs plain decode with greedy sampling (logits of the accepted tokens must match the S=1 path).

## Effort and risk
Estimated 1-2 weeks for a first version; the KDA state rollback and the S>1 lane are the risky parts. Payoff is conditional on the
text (code-like only). Recommended first step before building: replay existing generated sequences (`logs/perf/*.tokens`, the survey
outputs) through an n-gram/prompt-lookup drafter offline and measure acceptance a by domain (hours, no engine changes).
