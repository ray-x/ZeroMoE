#!/usr/bin/env python3
"""Offline acceptance of a model-free prompt-lookup drafter on the surveyed replies (word-level proxy for tokens).
For each reply: at every position the drafter looks for the most recent earlier occurrence of the last N words in prompt+reply and
proposes the K words that followed it; accepted = longest matching prefix. Reports mean accepted words per verification pass and
the share of passes that carried a draft, per domain."""
import collections, glob, re, sys
K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NGRAM = int(sys.argv[2]) if len(sys.argv) > 2 else 2
rows = []
for b in range(0, 10):
    ids = open('logs/perf/survey/batch%d.ids' % b).read().split()
    prompts = open('logs/perf/survey/p%d.txt' % b).read().split('\n')
    out = open('logs/perf/survey_b%d.stdout' % b, errors='replace').read()
    outs = [o.split('[Generation Complete]')[0] for o in out.split('Output:')[1:]]
    for pid, p, o in zip(ids, prompts, outs):
        rows.append((re.match(r'\d+_([a-z_+0-9]+?)__', pid).group(1), p.split(), o.split()))
def replay(prompt, reply):
    ctx = list(prompt); passes = drafted = acc_total = 0; t = 0
    while t < len(reply):
        passes += 1; draft = []
        if len(ctx) >= NGRAM:
            key = ctx[-NGRAM:]
            for s in range(len(ctx) - NGRAM - 1, -1, -1):
                if ctx[s:s + NGRAM] == key:
                    draft = ctx[s + NGRAM:s + NGRAM + K]; break
        a = 0
        while a < len(draft) and t + a < len(reply) and draft[a] == reply[t + a]: a += 1
        if draft: drafted += 1
        acc_total += a
        ctx.extend(reply[t:t + 1 + a]); t += 1 + a
    return passes, drafted, acc_total, len(reply)
bydom = collections.defaultdict(lambda: [0, 0, 0, 0])
for dom, p, r in rows:
    if len(r) < 8: continue
    ps, dr, ac, n = replay(p, r)
    x = bydom[dom]; x[0] += ps; x[1] += dr; x[2] += ac; x[3] += n
tot = [0, 0, 0, 0]
print('K=%d n-gram=%d, %d replies' % (K, NGRAM, len(rows)))
def line(name, x):
    ps, dr, ac, n = x
    # a pass carrying a draft is assumed to cost 1.5x a plain single-token pass
    print('%-16s words/pass %.2f  drafted %3.0f%% of passes  est. speed-up %.2fx' % (name, n / ps, 100.0 * dr / ps, n / (ps + 0.5 * dr)))
for dom in sorted(bydom):
    line(dom, bydom[dom])
    for i in range(4): tot[i] += bydom[dom][i]
line('ALL', tot)
