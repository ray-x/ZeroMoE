# Quant type vs capacity: modelled decode for GLM-5.3-Flash (2026-10-09)

`bench/quant_capacity_model.py` combines, per quant type: how many experts fit VRAM / DDR / SSD (bits per weight x GLM expert size
3 x 4096 x 1536), the share of routed traffic each tier serves (real per-expert usage counts, hottest experts placed first), the CPU
cost per expert (measured effective GB/s from `07-cpu-cost-by-quant.md`, Qwen-shaped benchmark) and the SSD miss cost.
`t_fixed` (dense/attention/routing per layer) and the SSD read rate are fitted to the measured AJ-IQ2_XXS decode runs
(DDR 24/32/48 GiB: 161/128/100 ms per token; avg expert 6.59 MB, dual-GPU spread config).

This is a MODEL, not a measurement. Caveats: static placement (a real LRU misses more); VRAM-hot routes cost nothing beyond `t_fixed`;
CPU GB/s comes from a different expert shape; `t_fixed` and SSD rate are fitted to a mixed-quant file and applied to uniform types;
the 3090-only tables reuse the fitted `t_fixed`. **Accuracy is not modelled** (bits per weight is only a rough proxy).
Each row is a hypothetical uniform-quant checkpoint, so the iq2_xxs row (4.9 MB/expert) is smaller than the real AJ file (6.59 MB).

fit: t_fixed=1.15 ms/layer, ssd=2.50 GB/s; AJ runs measured vs modelled ms/token:
   DDR 24 GiB: measured 161  modelled 161
   DDR 32 GiB: measured 128  modelled 128
   DDR 48 GiB: measured 100  modelled 101

### DDR 32 GiB, VRAM 28.9 GiB of experts, uniform quant per row (static placement by usage)

| type | bpw | MB/expert | in VRAM | in DDR | on SSD | route share VRAM/DDR/SSD | CPU GB/s | ms/token | tok/s | effective MoE/s |
|---|---|---|---|---|---|---|---|---|---|---|
| iq1_m | 1.75 | 4.1 | 7167 | 4929 | 0 | 89/11/0% | 11.3 | 62 | 16.1 | 5,400 |
| iq2_xxs | 2.06 | 4.9 | 6081 | 6015 | 0 | 83/17/0% | 11.8 | 72 | 13.9 | 4,656 |
| iq2_xs | 2.31 | 5.5 | 5424 | 6006 | 666 | 79/21/0% | 13.0 | 79 | 12.6 | 4,236 |
| q2_0 | 2.50 | 5.9 | 5017 | 5555 | 1524 | 76/23/1% | 14.8 | 88 | 11.4 | 3,835 |
| iq3_xxs | 3.06 | 7.2 | 4095 | 4535 | 3466 | 69/26/5% | 13.6 | 156 | 6.4 | 2,148 |
| q3_k | 3.44 | 8.1 | 3648 | 4040 | 4408 | 65/26/9% | 32.7 | 175 | 5.7 | 1,917 |
| iq4_xs | 4.25 | 10.0 | 2951 | 3267 | 5878 | 58/26/16% | 37.0 | 308 | 3.2 | 1,092 |
| q4_0 | 4.50 | 10.6 | 2787 | 3086 | 6223 | 56/25/18% | 37.5 | 354 | 2.8 | 949 |
| iq4_nl | 4.50 | 10.6 | 2787 | 3086 | 6223 | 56/25/18% | 35.1 | 357 | 2.8 | 942 |
| q4_k | 4.50 | 10.6 | 2787 | 3086 | 6223 | 56/25/18% | 41.8 | 350 | 2.9 | 961 |
| q5_k | 5.50 | 13.0 | 2280 | 2525 | 7291 | 51/24/26% | 41.7 | 550 | 1.8 | 611 |
| q6_k | 6.56 | 15.5 | 1911 | 2116 | 8069 | 46/22/32% | 45.4 | 776 | 1.3 | 433 |
| q8_0 | 8.50 | 20.1 | 1475 | 1633 | 8988 | 40/20/40% | 48.0 | 1224 | 0.8 | 275 |

### DDR 64 GiB, VRAM 28.9 GiB of experts, uniform quant per row (static placement by usage)

| type | bpw | MB/expert | in VRAM | in DDR | on SSD | route share VRAM/DDR/SSD | CPU GB/s | ms/token | tok/s | effective MoE/s |
|---|---|---|---|---|---|---|---|---|---|---|
| iq1_m | 1.75 | 4.1 | 7167 | 4929 | 0 | 89/11/0% | 11.3 | 62 | 16.1 | 5,400 |
| iq2_xxs | 2.06 | 4.9 | 6081 | 6015 | 0 | 83/17/0% | 11.8 | 72 | 13.9 | 4,656 |
| iq2_xs | 2.31 | 5.5 | 5424 | 6672 | 0 | 79/21/0% | 13.0 | 78 | 12.7 | 4,282 |
| q2_0 | 2.50 | 5.9 | 5017 | 7079 | 0 | 76/24/0% | 14.8 | 81 | 12.4 | 4,161 |
| iq3_xxs | 3.06 | 7.2 | 4095 | 8001 | 0 | 69/31/0% | 13.6 | 104 | 9.6 | 3,220 |
| q3_k | 3.44 | 8.1 | 3648 | 8080 | 368 | 65/35/0% | 32.7 | 78 | 12.8 | 4,309 |
| iq4_xs | 4.25 | 10.0 | 2951 | 6535 | 2610 | 58/39/3% | 37.0 | 126 | 7.9 | 2,669 |
| q4_0 | 4.50 | 10.6 | 2787 | 6172 | 3137 | 56/39/4% | 37.5 | 152 | 6.6 | 2,213 |
| iq4_nl | 4.50 | 10.6 | 2787 | 6172 | 3137 | 56/39/4% | 35.1 | 155 | 6.5 | 2,172 |
| q4_k | 4.50 | 10.6 | 2787 | 6172 | 3137 | 56/39/4% | 41.8 | 148 | 6.8 | 2,277 |
| q5_k | 5.50 | 13.0 | 2280 | 5050 | 4766 | 51/39/11% | 41.7 | 284 | 3.5 | 1,182 |
| q6_k | 6.56 | 15.5 | 1911 | 4232 | 5953 | 46/37/17% | 45.4 | 461 | 2.2 | 729 |
| q8_0 | 8.50 | 20.1 | 1475 | 3267 | 7354 | 40/34/26% | 48.0 | 840 | 1.2 | 400 |

### DDR 32 GiB, VRAM 15.8 GiB of experts, uniform quant per row (static placement by usage)

| type | bpw | MB/expert | in VRAM | in DDR | on SSD | route share VRAM/DDR/SSD | CPU GB/s | ms/token | tok/s | effective MoE/s |
|---|---|---|---|---|---|---|---|---|---|---|
| iq1_m | 1.75 | 4.1 | 3918 | 7936 | 242 | 67/33/0% | 11.3 | 89 | 11.3 | 3,788 |
| iq2_xxs | 2.06 | 4.9 | 3324 | 6734 | 2038 | 62/37/2% | 11.8 | 112 | 8.9 | 2,988 |
| iq2_xs | 2.31 | 5.5 | 2965 | 6006 | 3125 | 58/38/4% | 13.0 | 139 | 7.2 | 2,418 |
| q2_0 | 2.50 | 5.9 | 2743 | 5555 | 3798 | 56/38/7% | 14.8 | 159 | 6.3 | 2,110 |
| iq3_xxs | 3.06 | 7.2 | 2239 | 4535 | 5322 | 50/37/13% | 13.6 | 267 | 3.7 | 1,259 |
| q3_k | 3.44 | 8.1 | 1994 | 4040 | 6062 | 47/36/18% | 32.7 | 283 | 3.5 | 1,186 |
| iq4_xs | 4.25 | 10.0 | 1613 | 3267 | 7216 | 42/33/25% | 37.0 | 441 | 2.3 | 762 |
| q4_0 | 4.50 | 10.6 | 1523 | 3086 | 7487 | 41/32/27% | 37.5 | 494 | 2.0 | 680 |
| iq4_nl | 4.50 | 10.6 | 1523 | 3086 | 7487 | 41/32/27% | 35.1 | 498 | 2.0 | 675 |
| q4_k | 4.50 | 10.6 | 1523 | 3086 | 7487 | 41/32/27% | 41.8 | 488 | 2.0 | 688 |
| q5_k | 5.50 | 13.0 | 1246 | 2525 | 8325 | 36/29/34% | 41.7 | 711 | 1.4 | 472 |
| q6_k | 6.56 | 15.5 | 1044 | 2116 | 8936 | 33/27/40% | 45.4 | 957 | 1.0 | 351 |
| q8_0 | 8.50 | 20.1 | 806 | 1633 | 9657 | 29/24/48% | 48.0 | 1433 | 0.7 | 235 |

### DDR 64 GiB, VRAM 15.8 GiB of experts, uniform quant per row (static placement by usage)

| type | bpw | MB/expert | in VRAM | in DDR | on SSD | route share VRAM/DDR/SSD | CPU GB/s | ms/token | tok/s | effective MoE/s |
|---|---|---|---|---|---|---|---|---|---|---|
| iq1_m | 1.75 | 4.1 | 3918 | 8178 | 0 | 67/33/0% | 11.3 | 89 | 11.3 | 3,788 |
| iq2_xxs | 2.06 | 4.9 | 3324 | 8772 | 0 | 62/38/0% | 11.8 | 101 | 9.9 | 3,312 |
| iq2_xs | 2.31 | 5.5 | 2965 | 9131 | 0 | 58/42/0% | 13.0 | 107 | 9.3 | 3,128 |
| q2_0 | 2.50 | 5.9 | 2743 | 9353 | 0 | 56/44/0% | 14.8 | 108 | 9.3 | 3,123 |
| iq3_xxs | 3.06 | 7.2 | 2239 | 9070 | 787 | 50/50/0% | 13.6 | 139 | 7.2 | 2,412 |
| q3_k | 3.44 | 8.1 | 1994 | 8080 | 2022 | 47/51/2% | 32.7 | 110 | 9.1 | 3,041 |
| iq4_xs | 4.25 | 10.0 | 1613 | 6535 | 3948 | 42/51/7% | 37.0 | 197 | 5.1 | 1,709 |
| q4_0 | 4.50 | 10.6 | 1523 | 6172 | 4401 | 41/50/9% | 37.5 | 232 | 4.3 | 1,448 |
| iq4_nl | 4.50 | 10.6 | 1523 | 6172 | 4401 | 41/50/9% | 35.1 | 236 | 4.2 | 1,424 |
| q4_k | 4.50 | 10.6 | 1523 | 6172 | 4401 | 41/50/9% | 41.8 | 226 | 4.4 | 1,485 |
| q5_k | 5.50 | 13.0 | 1246 | 5050 | 5800 | 36/48/16% | 41.7 | 393 | 2.5 | 855 |
| q6_k | 6.56 | 15.5 | 1044 | 4232 | 6820 | 33/45/22% | 45.4 | 591 | 1.7 | 568 |
| q8_0 | 8.50 | 20.1 | 806 | 3267 | 8023 | 29/40/32% | 48.0 | 1000 | 1.0 | 336 |
