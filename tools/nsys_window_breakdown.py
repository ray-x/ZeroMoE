#!/usr/bin/env python3
"""Break each MTP verify window (NVTX range mtp.verify_window) of an nsys sqlite export into GPU busy time,
host waits in CUDA synchronisation calls, other CUDA API time, and host-only gaps (no CUDA call, no kernel:
CPU expert compute and scheduler work). Usage: analyze_window.py trace.sqlite"""
import sqlite3, sys, collections, statistics
db = sqlite3.connect(sys.argv[1])
def strings():
    return dict(db.execute("select id,value from StringIds").fetchall())
S = strings()
nv = db.execute("select start,end,text,textId,globalTid from NVTX_EVENTS where end is not null").fetchall()
def name(r): return r[2] if r[2] else S.get(r[3], "")
ranges = collections.defaultdict(list)
for r in nv: ranges[name(r)].append((r[0], r[1], r[4]))
wins = sorted(ranges["mtp.verify_window"])
print("windows:", len(wins), " other ranges:", {k: len(v) for k, v in ranges.items() if k.startswith(("win.", "promo", "mtp.", "lane", "dec."))})
kern = db.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
cpy = db.execute("select start,end from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
rt = db.execute("select start,end,nameId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME order by start").fetchall()
def clip(a, b, s, e): return max(0, min(b, e) - max(a, s))
def merged(iv, s, e):
    out = []; 
    for a, b in iv:
        a2, b2 = max(a, s), min(b, e)
        if b2 > a2: out.append((a2, b2))
    out.sort(); res = []
    for a, b in out:
        if res and a <= res[-1][1]: res[-1] = (res[-1][0], max(res[-1][1], b))
        else: res.append((a, b))
    return res
import bisect
kstarts = [k[0] for k in kern]
rstarts = [r[0] for r in rt]
rows = []
for (s, e, tid) in wins[5:-2]:
    kin = [k for k in kern[bisect.bisect_left(kstarts, s - 5_000_000): ] if k[0] < e]
    busy_iv = merged(kin, s, e)
    gpu_busy = sum(b - a for a, b in busy_iv)
    calls = [r for r in rt[bisect.bisect_left(rstarts, s - 5_000_000):] if r[0] < e and r[3] == tid]
    sync = other = 0; sync_iv = []; api_iv = []
    for a, b, nid, _ in calls:
        n = S.get(nid, "")
        d = clip(a, b, s, e)
        if "Synchronize" in n or "StreamQuery" in n or "EventQuery" in n: sync += d; sync_iv.append((max(a, s), min(b, e)))
        else: other += d; api_iv.append((max(a, s), min(b, e)))
    cpy_iv = merged([c for c in cpy[bisect.bisect_left([c[0] for c in cpy], s - 5_000_000):] if c[0] < e], s, e)
    def ulen(*ivs):
        flat = [x for iv in ivs for x in iv]
        return sum(b2 - a2 for a2, b2 in merged(flat, s, e))
    u_k = ulen(busy_iv); u_kc = ulen(busy_iv, cpy_iv); u_kcs = ulen(busy_iv, cpy_iv, sync_iv); u_all = ulen(busy_iv, cpy_iv, sync_iv, api_iv)
    covered = u_all
    excl = dict(x_gpu=u_k, x_copy=u_kc - u_k, x_sync=u_kcs - u_kc, x_api=u_all - u_kcs, x_host=(e - s) - u_all)
    sub = {k: sum(clip(a, b, s, e) for (a, b, t) in ranges.get(k, []) if t == tid) for k in ("win.submit", "win.sync", "win.readback", "promo.step", "mtp.commit_prefix", "mtp.sampling")}
    rows.append(dict(wall=e - s, gpu=gpu_busy, sync=sync, other=other, hostonly=(e - s) - covered, **excl, **sub))
n = len(rows)
def med(k): return statistics.median(r[k] for r in rows) / 1e6
print(f"\n{n} steady windows (nsys overhead included). Median per window, ms:")
print(f"  verify window wall            {med('wall'):6.2f}")
print(f"    win.submit (compute inline) {med('win.submit'):6.2f}    win.sync {med('win.sync'):6.2f}    win.readback {med('win.readback'):6.2f}")
print("  exclusive split of the window (sums to the wall):")
print(f"    GPU kernels running                 {med('x_gpu'):6.2f}")
print(f"    GPU copies only (no kernel)         {med('x_copy'):6.2f}")
print(f"    host waiting in sync/query, GPU idle{med('x_sync'):6.2f}   <- poll/wait for copies, events, CPU-split handoffs")
print(f"    host in other CUDA calls, GPU idle  {med('x_api'):6.2f}   <- launches / async copies / scheduler graph submit")
print(f"    host-only, GPU idle, no CUDA call   {med('x_host'):6.2f}   <- CPU expert compute + scheduler logic")
print(f"  inline sub-ranges: win.sync {med('win.sync'):.2f}  win.readback {med('win.readback'):.2f}  (win.submit {med('win.submit'):.2f} contains the compute)")
