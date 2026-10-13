#!/usr/bin/env python3
"""Per-session adaptive resident set, clean held-out replay. Initial set = top-N of a TRAIN-ONLY ranking; each test prompt starts
from it (fresh session). After every decode token up to U of the experts that just missed replace the least-recently-used
resident expert of the same layer. Prints cold routes/token for U=0 (static) and U>0, per prompt and on average."""
import collections, glob, sys
D='docs/ssd-offload/data/'
N=int(sys.argv[1]) if len(sys.argv)>1 else 4700
prompts=[]
for f in sorted(glob.glob('logs/perf/survey/b[0-9]*.trace')):
    last=None; cur=None
    for line in open(f):
        t=line.split()
        if not t: continue
        if t[0]=='P':
            pos=int(t[1])
            if last is None or pos!=last+1: cur=[]; prompts.append(cur)
            cur.append({}); last=pos
        elif t[0]=='R' and cur: cur[-1][int(t[1])]=[int(x) for x in t[2:]]
prompts=[p for p in prompts if len(p)>=8]
test=[p for i,p in enumerate(prompts) if i<40 and i%4==3]
train=[p for i,p in enumerate(prompts) if not (i<40 and i%4==3)]
c=collections.Counter()
for p in train:
    for tok in p:
        for L,ids in tok.items():
            for e in ids: c[(L,e)]+=1
base=[k for k,_ in c.most_common() if 3<=k[0]<=44][:N]
def run(p,U):
    res=collections.defaultdict(dict)
    for L,e in base: res[L][e]=0
    miss=routes=0; clock=0
    for tok in p:
        clock+=1; missed=[]
        for L,ids in tok.items():
            for e in ids:
                routes+=1
                if e in res[L]: res[L][e]=clock
                else: miss+=1; missed.append((L,e))
        for L,e in missed[:U]:
            if e in res[L] or not res[L]: continue
            del res[L][min(res[L],key=res[L].get)]; res[L][e]=clock
    return miss/len(p), routes/len(p)
print('test prompts',len(test),'train',len(train))
for U in (0,4,8,16,32):
    per=[run(p,U) for p in test]
    print('U=%2d cold routes/token mean %.1f (%.1f%%)  per prompt: %s'%(U,sum(m for m,_ in per)/len(per),100*sum(m for m,_ in per)/sum(r for _,r in per),' '.join('%.0f'%m for m,_ in per)))
