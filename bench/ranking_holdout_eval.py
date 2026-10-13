#!/usr/bin/env python3
"""Held-out evaluation of the expert ranking that decides the GPU-resident set.
Reads logs/perf/survey/b*.trace ("P pos" per decode token, "R layer ids..." per layer). Prompts are split where P jumps.
Splits prompts into TRAIN and TEST by index (every 4th prompt is held out); builds
  base   = glm53flash_usage_combined_old_plus_survey5 (what the engine ranks with today)
  refit  = base + TRAIN decode counts (scaled by --weight)
and reports the fraction of routes (layers 3-44) that fall outside the top-N set for each ranking on the TEST prompts."""
import argparse, collections, glob, re
ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=4700)           # resident experts (3090 cache + 5060 Ti)
ap.add_argument('--weight', type=float, default=1.0)
ap.add_argument('--test-first', type=int, default=0, help='hold out every Nth prompt only among the first K prompts; all later prompts train')
ap.add_argument('--holdout-every', type=int, default=4)
ap.add_argument('--write', help='write base + surveyed prompts (weight --weight) as a usage file')
ap.add_argument('--write-train-only', action='store_true', help='with --write: use only the training split (keeps the test prompts out of the file)')
a = ap.parse_args()
D = 'docs/ssd-offload/data/'
def load_usage(path):
    c = collections.Counter()
    for l in open(path):
        p = l.split()
        if len(p) == 3: c[(int(p[0]), int(p[1]))] += int(p[2])
    return c
prompts = []   # each: list of tokens, each token: dict layer -> ids
for f in sorted(glob.glob('logs/perf/survey/b[0-9]*.trace')):
    last = None; cur = None
    for line in open(f):
        t = line.split()
        if not t: continue
        if t[0] == 'P':
            pos = int(t[1])
            if last is None or pos != last + 1:
                cur = []; prompts.append(cur)
            cur.append({}); last = pos
        elif t[0] == 'R' and cur:
            cur[-1][int(t[1])] = [int(x) for x in t[2:]]
prompts = [p for p in prompts if len(p) >= 8]
is_test = lambda i: i % a.holdout_every == a.holdout_every - 1 and (a.test_first <= 0 or i < a.test_first)
test = [p for i, p in enumerate(prompts) if is_test(i)]
train = [p for i, p in enumerate(prompts) if not is_test(i)]
print('prompts %d (train %d, test %d); decode tokens train %d test %d' % (
    len(prompts), len(train), len(test), sum(map(len, train)), sum(map(len, test))))
def counts(ps):
    c = collections.Counter()
    for p in ps:
        for tok in p:
            for L, ids in tok.items():
                for e in ids: c[(L, e)] += 1
    return c
base = load_usage(D + 'glm53flash_usage_combined_old_plus_survey5.txt')
tr = counts(train)
tot_base = sum(base.values()); tot_tr = sum(tr.values()) or 1
def top(c): return set(k for k, _ in sorted(((k, v) for k, v in c.items() if 3 <= k[0] <= 44), key=lambda kv: -kv[1])[:a.n])
refit = collections.Counter(base)
for k, v in tr.items(): refit[k] += a.weight * v * tot_base / tot_tr   # same total mass as the base file
test_c = counts(test); tt = sum(test_c.values())
def cold(setx): return 100.0 * sum(v for k, v in test_c.items() if k not in setx) / tt
print('cold routes on the held-out prompts, top-%d resident:' % a.n)
print('  base ranking (today)        %.1f%%' % cold(top(base)))
print('  base + train prompts (refit) %.1f%%' % cold(top(refit)))
print('  train prompts only           %.1f%%' % cold(top(tr)))
print('  oracle (test prompts)        %.1f%%' % cold(top(test_c)))

if a.write:
    allc = counts(train if a.write_train_only else prompts); ta = sum(allc.values()) or 1
    out = collections.Counter(base)
    for k, v in allc.items(): out[k] += a.weight * v * tot_base / ta
    with open(a.write, 'w') as f:
        for (L, e), v in sorted(out.items()): f.write('%d %d %d\n' % (L, e, int(round(v))))
    print('wrote', a.write, len(out), 'entries from', len(train) if a.write_train_only else len(prompts), 'prompts')
