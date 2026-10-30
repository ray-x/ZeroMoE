#!/usr/bin/env python3
"""GLM decode timeline from an Nsight Systems export (--trace=cuda,nvtx,osrt --cuda-graph-trace=node:host-and-device, engine NVTX ranges
lane.run / lane.wait_ready per layer). usage: glm_nsys_lane_breakdown.py <dir with the .sqlite>"""
import sqlite3,glob,bisect,statistics as st
import sys
D=sys.argv[1].rstrip('/') + '/'   # directory holding the .nsys-rep exported to .sqlite (nsys export --type sqlite)
c=sqlite3.connect('file:'+glob.glob(D+'*.sqlite')[0]+'?mode=ro',uri=True)
sid=dict(c.execute('select id,value from StringIds'))
nv=c.execute('select start,end,text,textId from NVTX_EVENTS where end is not null order by start').fetchall()
run=[(s,e) for s,e,t,i in nv if (t or sid.get(i))=='lane.run']
wait=[(s,e) for s,e,t,i in nv if (t or sid.get(i))=='lane.wait_ready']
assert len(run)==len(wait)==4158
# pair each wait with the run that contains it
runs=[]; wi=0
for s,e in run:
    while wait[wi][0]<s: wi+=1
    runs.append((s,e,wait[wi][0],wait[wi][1]))
L=42; ntok=len(runs)//L
t0=runs[0][0]; t1=runs[-1][1]
def merge(iv):
    iv.sort(); out=[]
    for s,e in iv:
        if out and s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return out
K={}
for dev in (0,1):
    K[dev]=merge([[s,e] for s,e in c.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and end>=? and start<=?',(dev,t0,t1))])
starts={d:[s for s,e in K[d]] for d in K}
def busy(dev,a,b):
    iv=K[dev]; i=max(0,bisect.bisect_right(starts[dev],a)-1); tot=0
    while i<len(iv) and iv[i][0]<b:
        tot+=max(0,min(b,iv[i][1])-max(a,iv[i][0])); i+=1
    return tot
rows=[]
for k in range(ntok-1):
    ch=runs[k*L:(k+1)*L]; nxt=runs[(k+1)*L][0]
    wall=nxt-ch[0][0]
    w=sum(r[3]-r[2] for r in ch)                    # lane.wait_ready: CPU waits for GPU0 (attention/router)
    cpu=sum(r[1]-r[3] for r in ch)                  # CPU cold experts + output upload
    pre=sum(r[2]-r[0] for r in ch)                  # run start .. wait start (setup before waiting)
    gap=sum(ch[i+1][0]-ch[i][1] for i in range(L-1))+(nxt-ch[-1][1])   # between lane.run ranges (+ token tail)
    g0=busy(0,ch[0][0],nxt); g1=busy(1,ch[0][0],nxt)
    b0w=sum(busy(0,r[2],r[3]) for r in ch); b1w=sum(busy(1,r[2],r[3]) for r in ch)
    b0c=sum(busy(0,r[3],r[1]) for r in ch); b1c=sum(busy(1,r[3],r[1]) for r in ch)
    b0g=g0-sum(busy(0,r[0],r[1]) for r in ch); b1g=g1-sum(busy(1,r[0],r[1]) for r in ch)
    rows.append((wall,w,cpu,pre,gap,g0,g1,b0w,b1w,b0c,b1c,b0g,b1g))
sel=rows[20:90]
med=lambda i: st.median(r[i] for r in sel)/1e6
print(f'tokens analysed {len(sel)} (of {len(rows)}), steady window; times in ms per token, medians')
print(f'token wall (nsys-instrumented)      {med(0):7.1f}')
print(f'  CPU lane: wait for GPU0 result    {med(1):7.1f}   (lane.wait_ready, 42 layers)')
print(f'  CPU lane: cold experts + upload   {med(2):7.1f}   (lane.run minus wait)')
print(f'  CPU lane: setup before the wait   {med(3):7.1f}')
print(f'  between lane ranges (GPU part)    {med(4):7.1f}   (incl. token tail: LM head etc.)')
print(f'GPU0 3090 kernel busy per token     {med(5):7.1f}   ({100*med(5)/med(0):.0f}% of wall)')
print(f'GPU1 5060 Ti kernel busy per token  {med(6):7.1f}   ({100*med(6)/med(0):.0f}% of wall)')
print('overlap: share of each phase during which a GPU has a kernel running (union, so >1 stream counted once)')
print(f'  during CPU wait   : 3090 {100*med(7)/med(1):5.1f}%   5060 Ti {100*med(8)/med(1):5.1f}%')
print(f'  during CPU compute: 3090 {100*med(9)/med(2):5.1f}%   5060 Ti {100*med(10)/med(2):5.1f}%   <- parallelism with the CPU cold experts')
print(f'  between lane runs : 3090 {100*med(11)/med(4):5.1f}%   5060 Ti {100*med(12)/med(4):5.1f}%')
print(f'  3090 busy ms inside CPU compute window {med(9):.2f}; 5060 Ti {med(10):.2f}')
