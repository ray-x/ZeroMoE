#!/usr/bin/env python3
"""GLM decode timeline from an Nsight Systems export (--trace=cuda,nvtx,osrt --cuda-graph-trace=node:host-and-device, engine NVTX ranges
lane.run / lane.wait_ready per layer). usage: glm_nsys_gpu1_branch.py <dir with the .sqlite>"""
import sqlite3,glob,bisect,statistics as st
import sys
D=sys.argv[1].rstrip('/') + '/'   # directory holding the .nsys-rep exported to .sqlite (nsys export --type sqlite)
c=sqlite3.connect('file:'+glob.glob(D+'*.sqlite')[0]+'?mode=ro',uri=True)
sid=dict(c.execute('select id,value from StringIds'))
nv=c.execute('select start,end,text,textId from NVTX_EVENTS where end is not null order by start').fetchall()
run=[(s,e) for s,e,t,i in nv if (t or sid.get(i))=='lane.run']
t0,t1=run[0][0],run[-1][1]
ev1=sorted([(s,e) for s,e in c.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=1 and end>=? and start<=?',(t0,t1))]+
           [(s,e) for s,e in c.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=1 and end>=? and start<=?',(t0,t1))])
st1=[s for s,e in ev1]
d0=sorted([(s,e,k) for s,e,k in c.execute('select start,end,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=0 and end>=? and start<=?',(t0,t1))])
st0=[s for s,e,k in d0]
spans=[];gaps=[];pre=[];post=[]
for k in range(len(run)-1):
    a,b=run[k][1],run[k+1][0]
    i=bisect.bisect_left(st1,a)
    grp=[]
    while i<len(ev1) and ev1[i][0]<b: grp.append(ev1[i]); i+=1
    if not grp: continue
    f=grp[0][0]; l=max(e for s,e in grp)
    # feeding hop on the 3090 (D2H just before the first 5060 Ti event) and the return hop (3090 H2D right after the last)
    j=bisect.bisect_left(st0,f-30000); feed=f
    while j<len(d0) and d0[j][0]<f:
        if d0[j][2]==2 and f-d0[j][0]<30000: feed=min(feed,d0[j][0])
        j+=1
    j=bisect.bisect_left(st0,l-1000); end=l
    while j<len(d0) and d0[j][0]<l+60000:
        if d0[j][2]==1: end=max(end,d0[j][1]); break
        j+=1
    spans.append((end-feed)/1e3); gaps.append((b-a)/1e3); pre.append((feed-a)/1e3); post.append((b-end)/1e3)
n=len(spans)
print(f'{n} layer-steps. per layer: lane end -> next lane start {st.median(gaps):.0f} us; of which before the 5060 Ti hop starts {st.median(pre):.0f} us, 5060 Ti branch incl. both host hops {st.median(spans):.0f} us, after it {st.median(post):.0f} us')
per_tok=st.median(spans)*42/1000
print(f'5060 Ti branch on the critical path: {per_tok:.1f} ms per token (instrumented run: wall 87.9 ms -> {100*per_tok/87.9:.0f}%)')
print(f'scaled to the un-instrumented 70.8 ms token: ~{per_tok*70.8/87.9:.1f} ms')
