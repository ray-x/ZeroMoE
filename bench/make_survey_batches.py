#!/usr/bin/env python3
"""Pick prompts from tests/moe_usage_survey/prompts (excluding the ones already in the ranking), flatten each to one line and
write stdin files for the engine: logs/perf/survey/batch<i>.txt (+ batch<i>.ids). Stratified by category prefix."""
import os, re, sys, collections
root = 'tests/moe_usage_survey/prompts'
used = {1, 33, 65, 81, 105}            # the 5 survey prompts that went into glm53flash_usage_combined_old_plus_survey5
n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 40
per_batch = int(sys.argv[2]) if len(sys.argv) > 2 else 10
files = sorted(os.listdir(root))
bycat = collections.defaultdict(list)
for f in files:
    m = re.match(r'(\d+)_([a-z_+0-9]+?)__', f)
    if not m or int(m.group(1)) in used: continue
    bycat[m.group(2)].append(f)
picked = []
cats = sorted(bycat)
i = 0
while len(picked) < n_total and any(bycat.values()):
    c = cats[i % len(cats)]; i += 1
    if bycat[c]: picked.append(bycat[c].pop(0))
os.makedirs('logs/perf/survey', exist_ok=True)
for b in range(0, len(picked), per_batch):
    chunk = picked[b:b + per_batch]
    with open('logs/perf/survey/batch%d.txt' % (b // per_batch), 'w') as out:
        for f in chunk:
            text = ' '.join(open(os.path.join(root, f)).read().split())[:900]
            out.write(text + '\n')
        out.write(':q\n')
    with open('logs/perf/survey/batch%d.ids' % (b // per_batch), 'w') as ids:
        ids.write('\n'.join(chunk) + '\n')
print(len(picked), 'prompts;', collections.Counter(re.match(r'\d+_([a-z_+0-9]+?)__', f).group(1) for f in picked))
