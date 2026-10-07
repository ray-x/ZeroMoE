# GLM-5.3-Flash SSD offload inside the full graph: first end-to-end result (2026-10-08)

## What runs

`MOESTREAM_GLM5NEXT_OFFLOAD_FIRST_LAYER=11` (layers 11..44 offloaded; layers 3..10 stay VRAM-resident, as in the
baseline config) turns on, inside `src/ggml_glm5next_graph.cpp`:

* a per-projection virtual arena (`VirtualArena`, slot = expert id) bound as the CPU `MUL_MAT_ID` weights, so no id
  remapping is needed; the legacy 60 GiB host source arena is skipped for these layers;
* an LRU `ExpertCache` seeded at boot from the usage ranking (`..._OFFLOAD_USAGE`), populated from the offload store
  (`..._OFFLOAD_STORE`, aligned `preadv`) with the model shards (`..._OFFLOAD_MODEL`, first shard) as fallback;
* a miss node (`ggml_map_custom1`) in front of every offloaded layer's `MUL_MAT_ID`: it copies the routed ids, calls
  `ExpertResidency::ensure()` (decode) or `ensure_transient()` (prefill, bypasses the cache), and its output is the ids the
  matmuls consume, so ordering is guaranteed by data dependency. A load error aborts the process: no request is ever
  computed with a slot that does not hold the expert.

Knobs: `..._OFFLOAD_DDR_GIB` (expert DDR budget, default 40), `..._OFFLOAD_PINNED` (pinned fraction, default 0.5),
`..._OFFLOAD_THP` (1 = transparent huge pages for the arena, default off).

## Result (RTX 3090, expert DDR budget 32 GiB = 5,245 of 9,792 offloaded experts, 20.8 GB store, page cache dropped)

* Correctness: 96 generated tokens (greedy, same prompt) are byte-identical to the all-DDR full-graph baseline, in all 5
  offload runs, including with the model in page cache and with it dropped.
* Decode: baseline 145 ms/token in-graph (6.4 tok/s); offload ~220 ms/token with a warm page cache and ~273 ms/token
  end to end with a cold one (3.7 tok/s). About 25 expert misses per token, ~4.3 ms each when read from NVMe through
  buffered `pread`; the in-graph cost matches the sum of the loads (callback time = load time + 5%).
* Seeding 5,245 experts takes ~26-29 s at boot (sequential); it is now done in `glm5next_full_graph_bootstrap`, not in the
  first decode token.
* THP for the arena did not help (decode 2.10 vs 2.35 tok/s in the same conditions); left off.
* Prefill still runs the legacy path in both arms (the graph is decode-only here); the offload arm's prefill reads the
  engine's own RAM tier from SSD, which is why it is ~10 s slower with `host_ram_gb = 16`.

## Not done

* Loads are synchronous inside the miss node; overlapping them (prefetch from the next layer's pilot router, or issuing the
  reads of all 34 layers' predicted ids early) is the next lever: ~70-110 ms/token is pure load wait.
* The store holds only 3,173 experts; the other ~2,000 seeded and most misses are served from the model through buffered
  reads. A larger store (or O_DIRECT model reads) would make every miss one aligned `preadv`.
* Parallel seeding; per-layer capacity instead of one global LRU; graph-only policy for GLM5NEXT.

## Update: no whole-layer VRAM, spread hot experts + DDR/SSD for every layer (10 tok/s)

Config `configs/glm53flash_AJ_IQ2XXS_spread_offload.toml` (no `cpu_affinity_layers`, `vram_preload_prefix_layers = 0`,
`host_ram_gb = 16`), launcher `docs/ssd-offload/run_spread_offload.sh`. Every MoE layer (3..44) is offloaded
(`OFFLOAD_FIRST_LAYER=3`); VRAM holds each layer's hottest experts instead of whole layers.

Changes: row-aligned VRAM slot stride on every layer under the GLM graph (hot aliasing needs it, so the full-layer prefix is
no longer required); the miss node wraps only the cold-branch ids and the DDR seed skips VRAM-hot experts; model plane
reads and per-call expert loads run concurrently (miss 4.6 -> 2.9 ms); `MOESTREAM_GLM5NEXT_OFFLOAD_THP=1` (huge pages on
the arena, -22 ms/token of CPU matmul TLB cost); `MOESTREAM_GLM5NEXT_GPU1_HOT_GIB` puts the next-hottest experts of every
layer on the second GPU (third MUL_MAT_ID branch), assigned inside `configure_partial_experts` so every graph builder sees
the same split.

Steady-state decode (RTX 3090 + 5060 Ti, 8 CPU threads, page cache dropped before each run):

| configuration | ms/token | tok/s |
|---|---|---|
| original whole-layer baseline (8 layers in VRAM, 6 threads) | 143 | 7.0 |
| spread hot experts, 3090 only, no offload | 124 | 8.1 |
| + offload, 48 GiB DDR, 3090 only (THP) | 139 | 7.2 |
| + GPU1 hot 13 GiB, 48 GiB DDR (0 misses) | 104 | 9.6 |
| + GPU1 hot 14.2 GiB, 48 GiB DDR | 100.6 | 9.9 |
| same, 32 GiB DDR (7 misses/token) | 125 | 8.0 |
| same, 24 GiB DDR (17 misses/token) | 161 | 6.2 |

15 GiB is the GPU1 limit (15.5 OOMs). Output is coherent; it diverges from the pure-VRAM spread run at token 18 (different
summation order), and from the whole-layer baseline at token 1 (the legacy prefill uses a different hot set).
Below ~48 GiB of expert DDR the loads show up directly (2.9 ms each); hiding them behind a prefetch is the next lever.

### Logits check (teacher-forced on the whole-layer baseline's 96 tokens; `MOESTREAM_DECODE_LOGITS_DUMP`, `docs/ssd-offload/cmp_logits.py`)

| comparison | top-1 kept | step-0 max abs logit delta | median / max delta |
|---|---|---|---|
| spread (no offload) vs whole-layer baseline | 82/96 | 1.86 | 2.5 / 6.4 |
| spread + offload vs spread (identical prefill) | 96/96 | 0.000 | **0.000 / 0.000** |
| spread + offload + GPU1 hot 14.2 GiB vs whole-layer baseline | 87/96 | 1.86 | 2.2 / 6.7 |
| same vs spread (identical prefill) | 86/96 | 0.000 | 2.2 / 5.5 |
| spread + offload + GPU1 hot 0.3 GiB (48 experts) vs spread | - | 0.000 | ~0.7 / 1.5 (steps 1-11) |
| whole-layer baseline vs all-CPU prefill (step 0) | - | 1.08 | - |
| spread vs all-CPU prefill (step 0) | - | 1.74 | - |

* The DDR/SSD offload itself is bit-exact (zero logit delta over 96 steps).
* The deviation from the baseline starts in the legacy prefill: it grows with the number of layers whose experts run on the GPU
  (8 layers: 1.08 off the all-CPU prefill; 42 layers: 1.74). It is GPU-vs-CPU quantized-matmul arithmetic (q8_1 vs q8_K
  activation quantization), not an offload effect.
* The GPU1 hot branch changes decode logits by a delta that grows with the number of experts moved (0.7 at 48 experts,
  2.2 at 2,100+), the same direction and scale as the GPU-vs-CPU gap above; text stays coherent. This is consistent with
  arithmetic noise but is NOT proof of correctness against an independent reference (e.g. llama.cpp on the same tokens).

**GPU1 branch verified.** `MOESTREAM_GLM5NEXT_GPU1_HOT_ON_GPU0=1` puts the same hot1 expert set (0.6 GiB, 3090 cache shrunk to
make room) on the main GPU instead of the 5060 Ti, so the identical kernels run on a different device. Teacher-forced over 24
steps the logits are bit-identical (max abs delta 0.000, top-1 24/24). The map, mask, weight-ordering and accumulation of the
third branch are therefore correct; the delta against the CPU-cold placement is GPU-vs-CPU quantized-matmul arithmetic.
