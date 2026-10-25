# GLM-5.3 branch merged into master: notes for the Ling session

Date: 2026-10-20. Master was fast-forwarded from `7bda5da` to `9d9a8d8` (not pushed).
The GLM-5.3 decode work (`glm53-layered-speed`) now sits on top of the Ling 3.1 Flash work.
Two merges of master into the branch were needed (`5671ae7`, then `9d9a8d8`); the second had no conflicts.

## What landed

GLM-5.3 UD-IQ2_M decode on 3090 + 5060 Ti: 2.04-2.09 tok/s first-to-last (peak, short prompt), 1.92 on the Rates line,
up from 1.74-1.78 without the page-reuse eviction. Details: `docs/GLM5.3-baseline.md`, `bench/glm53/`.
Main changes: page-reusing eviction (`ZEROMOE_OFFLOAD_RETARGET`, on by default for any SSD-offload run, `=0` disables; see `docs/offload-retarget.md`), graph LM head, lane timers, and an
opt-in memory-guard floor override (`ZEROMOE_MEM_GUARD_FLOOR_MB`, default behaviour unchanged).

## Shared code the Ling session should know about

These files are used by both families. Review them if Ling behaves differently after you rebase.

| File | Change | Ling impact |
|---|---|---|
| `models/graph/full_graph_models.cpp` | Added `GlmDsaGraphModel` (`glm_dsa`) next to `Ling31FlashGraphModel` (`ling`); both registered | none expected |
| `models/glm/ggml_glm5next_graph_settings.cpp` | `graph_enabled_for`: GLM family is `glm_family(family) && env_enabled()`; Ling still returns `ling_full_graph_enabled()` | Ling gate untouched |
| `models/glm/ggml_glm5next_graph_settings.cpp` | `gpu_assist2_device(core_device)` now takes the core device and picks the other GPU (was hard-coded to device 1) | affects any caller; GLM only today |
| `models/glm/ggml_glm5next_graph_build.cpp` | attention dispatch is three-way: `dsa_plain` -> `build_glm_dsa_attn`, `bailing` -> `build_bailing_mla`, else `build_dsa` | Ling path unchanged |
| `models/glm/ggml_glm5next_graph_build_blocks.cpp` | `build_glm_dsa_attn` added; `build_bailing_mla` kept | none |
| `models/glm/ggml_glm5next_graph_{decode,run}.cpp` | `plain_residual = g.dsa_plain \|\| family == BailingMoE3`; `kv_cache_width` loops kept; null guards on `Rc` when `qk_rope > 0` | Ling uses the same `kv_cache_width` path as before |
| `core/engine_runner_session.cpp` | `glm5next_full_graph_set_vram_cache` now also for `GLM_DSA` (Ling already had it) | none |
| `memory/offload_store.hpp` | `VirtualArena::retarget` (mremap DONTUNMAP), default on when the offload store is active (`ZEROMOE_OFFLOAD_RETARGET=0` disables) | **on for any model that uses this store, including Ling if it does** |
| `memory/memory_guard.hpp` | opt-in `ZEROMOE_MEM_GUARD_FLOOR_MB` (>= 8192) | off by default |
| `models/glm/glm5next_graph_internal.hpp` | `kv_row` removed (master's `kv_cache_width` kept) | none |

## Verification

- Build: Release, `zeromoe` and `zeromoe_tests`, clean on the final tree.
- GLM, final tree (80 tokens, no swap, retarget on, 64 GiB pool): 2.04 tok/s first-to-last, Rates decode 1.92; text coherent
  ("How a Transformer Language Model Generates Text ..."). Log: `logs/split_ab_final.*` in the worktree.
- Ling smoke (first merge, before `7bda5da`), `configs/ling-3.1-flash-fullgraph.toml`, `--ngen 12 --temp 0`, prompt "Hi there":

  | Build | Decode | First-to-last |
  |---|---|---|
  | master (`10cc7ec`) | 1.05 tok/s | 0.99 tok/s |
  | merged | 1.06 tok/s | 1.00 tok/s |

  Neither build declined the graph.

## Open items for the Ling session

1. **First-token difference, unverified.** The master build printed `io...` and the merged build `>...` as the first
   output token. The captured logs interleave stdout with log lines, so I could not compare the full 12-token text.
   Please rerun both builds and diff the full output (or use `bench/ling/reference-audit/run.sh`).
2. The Ling smoke ran against the first merge only. The second merge (`7bda5da`, expert-transfer fix and perf audit) had no
   conflicts and the tree builds, but Ling was not rerun on it.
3. If you rebase a Ling branch onto this master, expect conflicts only in the files in the table above.
4. `ZEROMOE_OFFLOAD_RETARGET` is now on by default wherever the offload store is used. It was only tested on GLM. If Ling uses the store and misbehaves, set `ZEROMOE_OFFLOAD_RETARGET=0` to compare.
