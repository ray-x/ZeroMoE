#!/usr/bin/env python3
"""Offline study of next-layer cold-expert predictors on router activation dumps.

Inputs: dump dirs d0..dN (ZEROMOE_QWEN4EXP_PROBE_DUMP, L{l}.x.bin = layer-l FFN input, L{l}.s.bin = router
probabilities), the model's router weights, and per-prompt routing traces (ZEROMOE_QWEN4EXP_ROUTE_IDS_DUMP style
"token layer e:hot ...") that say which experts are in the static VRAM set. Predictor for layer l+1 uses only
x_l: pilot = softmax(W_{l+1} x_l). Scored on the cold evaluations (routed experts outside the hot set) of l+1.

usage: expert_predictor_variants.py <dump_root> <trace_dir> <model first shard> <prompt names, in dump order>
"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__) + '/../bench')
import gguf_min
root, tdir, model, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
D, E, K = 2560, 512, 10
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
head, tail = model.split('-00001-of-'); n = int(tail.split('.')[0])
W = {}
for i in range(1, n + 1):
    sh = f'{head}-{i:05d}-of-{n:05d}.gguf'
    for name, ent in gguf_min.parse(sh).items():
        if name.startswith('blk.') and name.endswith('.ffn_gate_inp.weight'):
            W[int(name.split('.')[1])] = torch.tensor(gguf_min.read_f32(sh, ent)).reshape(E, D).to(dev)
layers = sorted(l for l in W if l + 1 in W)

def hot_sets(prompt):
    hot = {}
    for line in open(f'{tdir}/{prompt}.ids'):
        f = line.split(); l = int(f[1])
        for x in f[2:]:
            e, h = x.split(':')
            if h == '1': hot.setdefault(l, set()).add(int(e))
    return hot

# metric accumulators: name -> [covered cold, cold total, hits (pred & actual), predicted, set size sum, samples]
acc = {}
def add(name, pred, actual, cold_actual):
    c = acc.setdefault(name, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    c[0] += (pred & cold_actual).sum().item(); c[1] += cold_actual.sum().item()
    c[2] += (pred & actual).sum().item(); c[3] += pred.sum().item(); c[4] += pred.sum().item(); c[5] += pred.shape[0]

for di, prompt in enumerate(names):
    hot = hot_sets(prompt)
    for l in layers:
        if l + 1 not in hot: continue
        x = torch.tensor(np.fromfile(f'{root}/d{di}/L{l}.x.bin', dtype=np.float32).reshape(-1, D)).to(dev)
        s = torch.tensor(np.fromfile(f'{root}/d{di}/L{l+1}.s.bin', dtype=np.float32).reshape(-1, E)).to(dev)
        T = min(300, x.shape[0]); x, s = x[-T:], s[-T:]           # decode rows only
        pilot = torch.softmax(x @ W[l + 1].T, dim=-1)
        actual = torch.zeros_like(s, dtype=torch.bool); actual.scatter_(1, s.topk(K, dim=-1).indices, True)
        hotv = torch.zeros(E, dtype=torch.bool, device=dev); hotv[list(hot[l + 1])] = True
        cold_actual = actual & ~hotv
        pilot_cold = pilot.masked_fill(hotv, -1.0)
        top10 = torch.zeros_like(actual); top10.scatter_(1, pilot.topk(K, dim=-1).indices, True)
        add('baseline: pilot top-10, cold part only', top10 & ~hotv, actual, cold_actual)
        for m in (1, 2, 3, 4, 6):
            P = torch.zeros_like(actual); P.scatter_(1, pilot_cold.topk(m, dim=-1).indices, True); P &= ~hotv
            add(f'idea 1: top-{m} cold experts by pilot prob', P, actual, cold_actual)
        for tau in (0.2, 0.1, 0.05, 0.02):
            add(f'idea 3: cold experts with pilot prob >= {tau}', (pilot > tau) & ~hotv, actual, cold_actual)
        # history: cold experts missed >= 2 times in the previous 32 tokens at this layer
        ca = cold_actual.float(); cs = torch.cat([torch.zeros(1, E, device=dev), ca.cumsum(0)])
        idx = torch.arange(T, device=dev); lo = (idx - 32).clamp(min=0)
        H = (cs[idx] - cs[lo]) >= 2
        add('history only: cold missed >=2x in last 32 tokens', H, actual, cold_actual)
        P3 = torch.zeros_like(actual); P3.scatter_(1, pilot_cold.topk(3, dim=-1).indices, True); P3 &= ~hotv
        add('idea 2: top-3 cold by pilot  UNION  history', P3 | H, actual, cold_actual)
        P6 = torch.zeros_like(actual); P6.scatter_(1, pilot_cold.topk(6, dim=-1).indices, True); P6 &= ~hotv
        add('idea 2: top-6 cold by pilot  UNION  history', P6 | H, actual, cold_actual)
print(f"{'predictor':56s} {'cold recall':>11s} {'precision':>10s} {'experts/token/layer':>20s}")
for name, c in acc.items():
    print(f"{name:56s} {100*c[0]/max(c[1],1):10.1f}% {100*c[2]/max(c[3],1):9.1f}% {c[4]/max(c[5],1):20.2f}")
