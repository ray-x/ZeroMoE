#!/usr/bin/env python3
"""Build stdin prompt files for a GLM route survey from tests/moe_usage_survey/prompts.
Prompts already in the shipped ranking (ids 1, 33, 65, 81, 105) are excluded. Selection is round-robin over the categories, so
every domain is represented; every `holdout_every`-th picked prompt goes to the TEST file, the rest to TRAIN. One prompt per line.
  bench/make_survey_sets.py <outdir> [n_total=100] [holdout_every=4]    writes train.txt, test.txt, train.ids, test.ids"""
import collections, os, re, sys
out = sys.argv[1]
n_total = int(sys.argv[2]) if len(sys.argv) > 2 else 100
every = int(sys.argv[3]) if len(sys.argv) > 3 else 4
root = 'tests/moe_usage_survey/prompts'
used = {1, 33, 65, 81, 105}
bycat = collections.defaultdict(list)
for f in sorted(os.listdir(root)):
    m = re.match(r'(\d+)_([a-z_+0-9]+?)__', f)
    if m and int(m.group(1)) not in used:
        bycat[m.group(2)].append(f)
cats = sorted(bycat)
picked, i = [], 0
while len(picked) < n_total and any(bycat.values()):
    c = cats[i % len(cats)]; i += 1
    if bycat[c]: picked.append(bycat[c].pop(0))
os.makedirs(out, exist_ok=True)
split = {'train': [], 'test': []}
for k, f in enumerate(picked):
    split['test' if k % every == every - 1 else 'train'].append(f)
for name, files in split.items():
    with open(f'{out}/{name}.txt', 'w') as t, open(f'{out}/{name}.ids', 'w') as ids:
        for f in files:
            t.write(' '.join(open(os.path.join(root, f)).read().split())[:900] + '\n')
            ids.write(f + '\n')
print(len(picked), 'prompts:', len(split['train']), 'train,', len(split['test']), 'test;', len(cats), 'categories')
