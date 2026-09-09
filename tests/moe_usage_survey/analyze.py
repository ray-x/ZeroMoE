#!/usr/bin/env python3
"""Real analysis of the survey's per-(layer,expert) usage data: Option A
(per-layer coverage threshold) and Option B (global ranking with floor).
Read-only, no engine/model touched."""
import collections

USAGE_FILE = "/home/ray/Projects/MoEStream/tests/moe_usage_survey/moestream_usage_survey"

# Real per-expert byte sizes, confirmed this session (gguf_tool dma / probe_slab).
GATE_UP_CURRENT = 3244032   # IQ2_XXS, bytes/expert (layer 8 is IQ2_S=3936KB but
                             # treated as IQ2_XXS-equivalent below for simplicity --
                             # 1/75 layers, negligible to the aggregate).
GATE_UP_IQ1S = 2457600
DOWN_UNCHANGED = 4816896    # IQ3_XXS/IQ4_XS mix, unaffected either way

n_layers, n_experts = 75, 256
counts = {}  # (layer, expert) -> count
with open(USAGE_FILE) as f:
    for line in f:
        p = line.split()
        if len(p) == 3:
            counts[(int(p[0]), int(p[1]))] = int(p[2])

layers = sorted(set(l for l, e in counts))
assert len(layers) == n_layers, len(layers)

# ---- Option A: per-layer coverage threshold ----
def option_a(threshold):
    per_layer_k = {}
    for l in layers:
        row = sorted((counts[(l, e)] for e in range(n_experts)), reverse=True)
        total = sum(row)
        cum, k = 0, 0
        for c in row:
            cum += c
            k += 1
            if cum >= threshold * total:
                break
        per_layer_k[l] = k
    return per_layer_k

a80 = option_a(0.80)
a90 = option_a(0.90)

def summarize_k(per_layer_k, label):
    ks = list(per_layer_k.values())
    ks_sorted = sorted(per_layer_k.items(), key=lambda kv: kv[1])
    print(f"\n=== Option A ({label}) — per-layer K (experts kept at original precision) ===")
    print(f"  min K={ks_sorted[0][1]} (layer {ks_sorted[0][0]}), "
          f"max K={ks_sorted[-1][1]} (layer {ks_sorted[-1][0]}), "
          f"avg={sum(ks)/len(ks):.1f}, median={sorted(ks)[len(ks)//2]}")
    print(f"  total kept-original across all layers: {sum(ks)} of {n_layers*n_experts} ({100*sum(ks)/(n_layers*n_experts):.1f}%)")
    print(f"  5 most 'hot-minority' layers (smallest K):")
    for l, k in ks_sorted[:5]:
        print(f"    layer={l:3d}  K={k:3d}/256 ({100*k/256:.1f}%)")
    print(f"  5 flattest layers (largest K):")
    for l, k in ks_sorted[-5:]:
        print(f"    layer={l:3d}  K={k:3d}/256 ({100*k/256:.1f}%)")
    return sum(ks)

kept_80 = summarize_k(a80, "80% coverage")
kept_90 = summarize_k(a90, "90% coverage")

# ---- Option B: global ranking with a per-layer floor ----
def option_b(n_kept_total, floor_per_layer=1):
    # Floor first.
    kept = set()
    for l in layers:
        row = sorted(((counts[(l, e)], e) for e in range(n_experts)), reverse=True)
        for c, e in row[:floor_per_layer]:
            kept.add((l, e))
    remaining_budget = n_kept_total - len(kept)
    # Global ranking of everything not already kept.
    pool = [((l, e), counts[(l, e)]) for l in layers for e in range(n_experts) if (l, e) not in kept]
    pool.sort(key=lambda x: -x[1])
    for (l, e), c in pool[:max(0, remaining_budget)]:
        kept.add((l, e))
    per_layer_k = collections.Counter(l for l, e in kept)
    return per_layer_k, kept

b_kept, b_set = option_b(kept_80, floor_per_layer=1)
print(f"\n=== Option B (global ranking, floor=1/layer, budget matched to A@80% = {kept_80}) ===")
ks = [b_kept.get(l, 0) for l in layers]
ks_sorted = sorted(((l, b_kept.get(l, 0)) for l in layers), key=lambda kv: kv[1])
print(f"  min K={ks_sorted[0][1]} (layer {ks_sorted[0][0]}), max K={ks_sorted[-1][1]} (layer {ks_sorted[-1][0]}), avg={sum(ks)/len(ks):.1f}")
print(f"  layers reduced to the bare floor (K=1): {sum(1 for k in ks if k == 1)} of {n_layers}")

# Overlap between A@80% and B
a80_set = set()
for l in layers:
    row = sorted(((counts[(l, e)], e) for e in range(n_experts)), reverse=True)
    for c, e in row[:a80[l]]:
        a80_set.add((l, e))
overlap = len(a80_set & b_set)
print(f"\n  A@80% vs B: {overlap} of {len(a80_set)} kept-original experts agree ({100*overlap/len(a80_set):.1f}% overlap)")

# ---- Real savings estimate under A@80% ----
total_experts = n_layers * n_experts
downgraded = total_experts - kept_80
bytes_before = total_experts * (2 * GATE_UP_CURRENT + DOWN_UNCHANGED)
bytes_after = kept_80 * (2 * GATE_UP_CURRENT + DOWN_UNCHANGED) + downgraded * (2 * GATE_UP_IQ1S + DOWN_UNCHANGED)
print(f"\n=== Real savings estimate, Option A @ 80% coverage ===")
print(f"  {downgraded} of {total_experts} experts ({100*downgraded/total_experts:.1f}%) would move to IQ1_S gate/up")
print(f"  model MoE weight bytes: {bytes_before/1e9:.2f} GB -> {bytes_after/1e9:.2f} GB ({100*(1-bytes_after/bytes_before):.1f}% reduction)")
