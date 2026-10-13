#!/usr/bin/env python3
"""CPU vs GPU-upload decision for a cold routed expert (analytical, from measured inputs).
Inputs (all measured on this box): CPU expert cost per quant (docs/ssd-offload/07), PCIe H2D bandwidth 12.2 GB/s (3090) / 6.9 GB/s
(5060 Ti), DMA issue cost ~7 us, staging overhead of the in-graph pilot (0.5 ms/layer of extra scheduler work, +3.5% CPU while the DMA runs).
A cold expert can be (a) computed on the CPU from DDR, or (b) predicted, copied to VRAM (second lane) and computed on the GPU.
Per layer with c cold routes, K staged with precision p:  saved = K*p*t_cpu ;  cost = fixed + bounce(K) + wasted-copy CPU share."""
QUANT = {  # name: (MB/expert, us/expert on the CPU kernel, 8 threads)  -- docs/ssd-offload/07-cpu-cost-by-quant.md
    'q4_0': (2.76, 74), 'q3_k': (2.53, 77), 'q4_k': (3.32, 79), 'iq4_xs': (3.13, 85), 'q2_0': (1.38, 94), 'q6_k': (4.84, 106),
    'q8_0': (5.22, 109), 'iq2_xxs': (1.52, 129), 'iq2_xs': (1.70, 131), 'iq3_xxs': (2.26, 166),
    'GLM iq2_xxs (4.87MB, D4096 F1536)': (4.87, 390),
}
LINKS = {'3090 (PCIe4 x16)': 12.2, '5060 Ti (PCIe3)': 6.9}
FIXED_MS_PER_LAYER = 0.5      # measured extra graph/scheduler work of the staging nodes
print('%-36s %7s %8s | %s' % ('quant', 'MB', 'cpu us', '  '.join('%s: dma us / ratio cpu:dma' % k for k in LINKS)))
for q, (mb, cpu) in QUANT.items():
    cells = []
    for name, gbs in LINKS.items():
        dma = mb * 1e3 / gbs + 7        # us
        cells.append('%6.0f / %.2f' % (dma, cpu / dma))
    print('%-36s %7.2f %8d | %s' % (q, mb, cpu, '            '.join(cells)))
print('\nbreak-even staged experts per layer on the 3090 (pinned, no bounce) to repay the fixed %.1f ms/layer: K*p*t_cpu > fixed + 0.035*cpu_layer' % FIXED_MS_PER_LAYER)
for q in ('GLM iq2_xxs (4.87MB, D4096 F1536)', 'iq3_xxs', 'q2_0', 'q4_k'):
    mb, cpu = QUANT[q]
    for c in (3, 5):               # cold routes per layer (GLM telemetry: 5.1)
        for p in (0.9, 0.65):
            cpu_layer = c * cpu / 1000
            need = (FIXED_MS_PER_LAYER + 0.035 * cpu_layer) / (p * cpu / 1000)
            print('  %-34s c=%d p=%.2f: need K >= %.1f (cpu layer %.2f ms)%s' % (q, c, p, need, cpu_layer, '  -> never repays' if need > c else ''))
