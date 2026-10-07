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
