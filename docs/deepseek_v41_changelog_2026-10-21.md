# DeepSeek V4.1 Flash: changelog of the dsv41-on-master merge (2026-10-21)

12 commits on top of the refactored master (c03d677), merged with master 04e2f60. Hardware: RTX 3090 +
RTX 5060 Ti, Release build (`-O3 -DNDEBUG`), model `DeepSeek-V4.1-Flash-Q2_K` (40 layers, 384 experts,
top-6, expert = 12.81 MB, routed experts about 196.8 GB of the 265 GB model). The goal of 10 tok/s was
NOT reached; honest steady-state decode is about 4.0-4.5 tok/s.

## Fixes (correctness)
- Route-callback crash: the per-layer cache-site table was a shared global (260a2d9).
- SwiGLU clamp: V4.1 used the project-default product clamp. It now uses the reference fused clamp
  (min(gate, limit), clamp(up, +-limit), silu(gate)*up); graph shared experts previously had no clamp.
  Output text changes versus older builds. Not verified against a reference run (no llama.cpp here loads
  arch `deepseek41`).
- Mirror coverage: a sparse mirror read holes as zeros; reads now go through a per-expert coverage guard.

## Performance features (default behaviour changed)
- AVX-512 weighted sum in per-head attention and parallel indexer scoring (e3eed9d): decode +53-120%
  (the scalar loop was the top bucket, 206 of 430 ms/token).
- GPU hyper-connections in decode, default on, `MOESTREAM_DEEPSEEK_V41_DECODE_GPU_HC=0` opts out
  (260a2d9): +21% decode (hc_engram 59 -> about 2 ms/token). Text differs from the host path at about
  token 7 through a near-tie top-k flip, not a bug.
- `pin_global_tiers` (EngineConfig, default off, on in the DeepSeek config): VRAM-LRU and DDR tiers use
  one global ranking. Prefill +15% on a short prompt, decode unchanged.
- Config `pin_ram_fraction = 1.0` (27b81a5, DDR fill at startup): decode +8-11%.
- Config `ram_pinned_budget_gb = 40` (8394658): decode +5-8%.
- Release build: +38% decode versus an ad-hoc build without `CMAKE_BUILD_TYPE` (no `-O`).

## New opt-in switches (all default off, output-neutral unless noted)
| switch | commit | measured |
|---|---|---|
| `mirror_model_path` + `mirror_coverage_file` (down-only sparse mirror on a 2nd NVMe; built by `bench/dsv41_pack_cold_mirror.py`) | dba16ce | disk time -10..-31%, decode about +2% |
| `decode_promote_all_misses` | 5650d2f | +1.6% decode |
| `MOESTREAM_DSV41_PILOT_K=k` (next-layer router prefetch) | f82d3ac | disk -30%, decode +2-4% |
| `MOESTREAM_DSV41_CACHE_BONUS_VRAM/_RAM` (cache-aware expert selection) | 0737a17 | 3.62 -> 5.9 (b=0.40) .. 7.64 tok/s on one teacher-forced text. **Changes the output; quality cost not yet measured.** |
| `MOESTREAM_ROUTE_LOG`, `MOESTREAM_ROUTER_IN_LOG`, `MOESTREAM_TEACHER_FORCE_NLL` | 00150ef, f82d3ac, 0737a17 | instrumentation |

Tools: `bench/dsv41_graph_profile.py` (`--quantized-graph`, `--prompt-file`), `bench/dsv41_mirror_io_bench.py`,
`bench/dsv41_pilot_replay.py`, usage-survey config.

## Decode time breakdown (about 230 ms/token)
SSD reads (cold misses) 55-70 ms; H2D expert uploads about 75 ms (PCIe Gen3, about 11 GB/s); attention
plus dense about 45 ms; remainder (about 40 ms) is sync and unattributed. Cold picks are about 10-11% of
all picks (about 25-27 per token).

## Findings worth keeping
- Pilot router (layer N+1's `ffn_gate_inp` on layer N's router input) names 9.7/18.8/28.2/37.9/46.1/52.8%
  of cold picks at top-1..6 (62.6% top-8, 73.4% top-12). History-based prediction names about 0.2%.
- Down-only mirror IO bench: split read 1.58-1.61x per expert. Whole-expert alternating placement is
  1.00x at k=1 and about 1.8x at k>=2, but 80-85% of cold layer-steps have exactly one miss, so it gains
  little. The hybrid layout is not built.
- The cache-backed decode graph (`MOESTREAM_DEEPSEEK_V41_CACHE_MOE_DECODE=1`) is worse (3.3-3.75 tok/s).
- Prefill is storage-bound (about 29 tok/s on 817 tokens, measured before the Release build and DDR fill;
  not re-measured).

## Merge with master
Master added a whole-expert sparse mirror (`mirror_subset.tsv`, `resolve_replica`). Both mirror modes now
coexist: the down-only coverage guard runs first, then master's whole-expert resolve. Pick one mode per
run by which mirror files exist. Conflicts were `core/engine.hpp` and `core/engine_runner_cache.cpp`
(master's `glm_graph_in_use` kept, DeepSeek row-alignment condition kept).

## Verification on hardware (2026-10-21, temp 0, 64 tokens, DNS prompt, single cold runs)
| run | tokens vs pre-merge binary | decode | prefill |
|---|---|---|---|
| pre-merge binary | reference | 3.05 tok/s | 3.14 tok/s |
| merged binary | identical | 2.90 tok/s | 2.98 tok/s |
| merged + down-only mirror (7 shards, 9,469 experts; 13% of disk bytes from the mirror) | identical | 3.14 tok/s | 3.27 tok/s |

The 5% gap between the first two rows is within single-run noise; no repeated runs were done.
Not run on hardware: master's `mirror_subset.tsv` mode, a long steady-state decode, a large quality
evaluation of cache-aware selection (`/var/tmp/dsv41_env/ab/run_qual76.sh`, about 60 min).
