# Option-B VRAM budget, legacy cache removal and expert-pool sizing (2026-10-05)

Qwen3.8-Flash-Next **Q2_0** (`/data/sdb/models/qwen3.8-flash-next/Q2_0`, verified with `gguf_tool verify`), RTX 3090 24 GiB (220 W power limit),
Xeon W-2235, 10 CPU threads, single-GPU full-graph decode (`MOESTREAM_QWEN4EXP_FULL_GRAPH=1`, `..._DECODE=1`, `GRAPH_PRELOAD_LAYERS=0`),
engine `../moestream-build-upstream-master/moestream` built from this tree. Decode speed = mean of `TOKEN-LOOP-DIAG iter_total`, steps >= 3.
Related: `docs/handoff-2026-10-03-option-b-global-pool.md` (global pool and the teacher-forced numerics), `docs/runtime-settings.md`.

## 1. Design policy that drove this work

Qwen4Exp has **no layered decode** and, going forward, every supported model must have a full-graph decode unless exempted. A failed graph
build is a failed request, not a silent fallback (see memory note `design-policy-full-graph-only-no-layered`). Other families (GLM5Next,
DeepSeek-V4, ...) still use the shared layered machinery; do not delete it for them.

## 2. Code state

| Commit / state | What |
|---|---|
| `99fd1ac` | Option-B global expert-pool selection, off by default (`qwen4exp_option_b_global_top_k`, `..._global_floor_per_layer`) |
| `4d97b26` | Qwen4Exp full graph: legacy VRAM expert cache and active-layer area are never created when the decode graph is on |
| `2138f09`, `2cc77bf` | Fixes: partial staging had mis-ordered `main.cpp` (use before declaration) and mis-placed the unit test in `tests/test_components.cpp`; both verified by compiling an exported clean HEAD tree |
| **uncommitted** | `MOESTREAM_QWEN4EXP_GRAPH_ALLOC_FLOOR_MIB` override in `src/ggml_qwen4exp_graph.cpp` (default unchanged); `bench/option_b_vram_probe.sh`, `bench/option_b_vram_stress.sh`, `bench/option_b_survey_coverage.py`, this document |

## 3. Findings

**F1. The legacy VRAM expert cache was 6.44 GiB of dead VRAM.** Config A (`configs/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml`) had
`moe_cache_gb = 11.5` (inherited from the older non-graph config); the engine capped it to the leftover VRAM, a 4,650-slot "compact expert VRAM
cache" (6.43796 GB; configured as a pinned-by-ranking tier plus an LRU part, `pin_vram_lru_fraction = 0.1`: the exact slot split was not checked in the code). Engine logs of that run: `[PIN] Preloaded 0 VRAM-pinned + 0
VRAM-unpinned`, `[VRAM-SLOTS] 0 / 4650 slots filled` after prefill and at shutdown, `Expert cache: 0 hit / 0 miss`. The only reference to it
that I found in the Qwen graph code is the old layer-13 staging proof. Disabling it (now automatic, `4d97b26`): GPU used at decode **20.65 -> 14.64 GiB**, free **3.35 -> 9.36 GiB**,
decode 29.1 -> 30.4 tok/s (single runs, i.e. unchanged), 402-token prefill 12.5 s vs 12.6 s with identical output, plain 22/26 still 26.4 tok/s. Accounting before
the change: 10.84 arena + 3.10 dense + 6.44 legacy cache + ~0.3 context/KV/state = 20.65.

**F2. The global ranking by itself does nothing; including all 48 layers helps a little.** 7 order-balanced reps, cache off, K = 8,421 in every row:
A (right-sized, 44 layers) 31.00 tok/s; V1 (global ranking, same 44 layers) 31.06 (+0.2%); B (global, all 48 layers) 31.49 (**+1.6%**, 7/7 reps > A, B > V1 in 7/7).
The gain comes from giving layers 0, 1, 2 and 22 hot experts (48/0 instead of 44/4). Earlier run with the legacy cache still on: B +1.1% (31.11 vs 30.78).

**F3. Spending the freed VRAM on more hot experts is the big lever.** One expert slab is 1,382,400 B (1.318 MiB); the 6.0 GiB freed is ~4,660 experts. Global pool over all 48 layers,
`bench/option_b_vram_probe.sh` (200-token chat decode):

| K | % of 24,576 | arena | steady free VRAM | decode tok/s | survey-expected coverage |
|---|---|---|---|---|---|
| 8,421 (current) | 34% | 10.84 GiB | 9.36 GiB | ~30.4 (A) | 76.3% |
| 13,100 | 53% | 16.87 GiB | 3.34 GiB | 35.65 | 90.5% |
| 14,500 | 59% | 18.67 GiB | 1.70 GiB | 38.99 | 93.2% |
| 14,554 | 59% | 18.74 GiB | 1.63 GiB | 39.11 | 93.3% |
| 14,746 | 60% | 18.99 GiB | 1.38 GiB (needs the floor override, F4) | 39.03 | 93.6% |

Single runs; run-to-run spread is a few percent. Coverage is predicted from the survey (`bench/option_b_survey_coverage.py`), not measured live. Returns flatten above ~14,450.

**F4. Admission floor and the all-CPU fallback.** The graph allocator refuses to reserve when free VRAM < 1.00 GiB. At K = 14,746 the decode-graph build saw 0.98 GiB free:
`refusing graph allocator reserve: device 0 free=0.98 GiB < 1.00 GiB safety floor ... falling back to the legacy path`, then `graph construction declined; layered decode
remains active`: the run decoded **all-CPU at ~835 ms/token** (the legacy cache was gone, F1). By arithmetic K <= ~14,729 passes the default floor. The code comment says the measured S=1
decode workspace is ~0.13 GiB, so the floor is conservative for decode; `MOESTREAM_QWEN4EXP_GRAPH_ALLOC_FLOOR_MIB=512` (opt-in, 64..4096) lets K = 14,746 build and decode (steady free 1.38 GiB, min 1.22).
This is exactly the silent-slow-fallback the full-graph-only policy forbids: stage 2 should turn it into a hard error.

**F5. Stress sessions (`bench/option_b_vram_stress.sh`: one process, 5 requests SHORT/LONG/SHORT/LONG/SHORT = 28/402/28/402/28 tokens, 48 new tokens each).**

| K | arena | min free VRAM | fallback / errors | 28-token req tok/s | 402-token req tok/s |
|---|---|---|---|---|---|
| 13,100 | 16.87 GiB | 2.89 GiB | none | 39.4 | 33.1 |
| 14,458 | 18.62 GiB | 1.56 GiB | none | 41.9 | 38.7 |
| 14,554 | 18.74 GiB | 1.44 GiB | none | 41.6 | 37.2 |
| 14,746 (floor 512 MiB) | 18.99 GiB | 1.19 GiB | none | 42.1 | 39.0 |

(14,554 re-run with the committed script: min free 1.44 GiB, decode 39.1/37.8/41.0/39.0/41.5.) All requests completed and GPU memory returned to idle at exit.
Prefill of the 402-token prompt: 12.3 s at every K. The engine **releases the expert arena before a prefill that needs more workspace than is free** (policy estimate: 1.0 GiB floor
+ 0.5 GiB + 7.5 MiB per token, e.g. "S=402 needs ~4.44 GiB, only 1.19 GiB free") and rebuilds it for decode (3.5-3.9 s). At K = 14,554 even a 17-token prompt triggered the release on
the first request; later requests showed fewer release/rebuild messages (1 release in 5 requests at K >= 14,554, 2 at K = 13,100) and I did not establish why. A large K therefore adds
a one-off ~4 s at the first request after startup; whether it repeats under other request patterns is untested.

**F6. Numerics.** Teacher-forced logits vs config A (`bench/option_b_global_pool_check.py`): determinism control bit-identical; on 4 ordinary prompts the global layouts have mean KL
0.0007-0.0009 and ~99% top-1 agreement, indistinguishable from the per-layer layout. One identifier-copy prompt diverged in every layout (near-tie flips). Details: handoff doc section 8.
Not yet run at K >= 13,100.

## 4. How to reproduce

Prerequisites: idle 3090 (the scripts wait for it), quiet CPU, `../moestream-build-upstream-master/moestream` built from this tree (`make -j6 moestream` in that directory),
model verified: `../moestream-build-upstream-master/gguf_tool verify /data/sdb/models/qwen3.8-flash-next/Q2_0`. Run from the repo root. Each script writes under `logs/` and exits non-zero on fallback/error.

```bash
python3 bench/option_b_survey_coverage.py                       # F3 sizing and coverage table (needs no GPU)
bench/option_b_vram_probe.sh 14554                              # F3: steady VRAM + decode tok/s for K=14554 (~1 min)
MOESTREAM_QWEN4EXP_GRAPH_ALLOC_FLOOR_MIB=512 bench/option_b_vram_probe.sh 14746   # F4: K above the default floor
bench/option_b_vram_stress.sh 14554                             # F5: 5-request session (~3 min)
bench/option_b_global_pool_ab.sh 0 7 128                        # F2: A/B/C/D, 7 order-balanced reps (A = right-sized baseline)
python3 bench/option_b_global_pool_check.py --prompt-file <single-line prompt> --out-dir logs/<dir> --ngen 64   # F6
```
What the numbers are: `steady used/free` = nvidia-smi `memory.used` sampled every 250 ms over the process, steady = samples above 90% of the peak, free = 24,576 MiB minus that;
`min free` = 24,576 MiB minus the single highest sample; `decode tok/s` = 1000 / mean(`iter_total`, steps >= 3), per request in the stress script (steps restart at 0).
Expected on this box (within a few %): probe K=14,554 free 1.5-1.7 GiB and 39 tok/s; stress K=14,554 min free 1.44 GiB, prefill 402 tokens 12.3 s, no fallback.
F1 can be re-checked by reading `[VRAM-SLOTS]` and `Expert cache:` lines in any engine log of an older binary, or by running the config with `qwen4exp_option_b_disable_legacy_cache` off on a pre-`4d97b26` build.
Prompts: SHORT = "Explain briefly the key concepts of neural networks and gradient descent in plain English." (28 tokens with the chat template); LONG = first 1,100 characters of
`docs/runtime-settings.md` at commit `b5e97c9` with newlines turned into spaces + " Question: summarize the text above in two sentences." (402 tokens; the stress script rebuilds it into
`logs/option_b_global_pool_check_prompt.txt` if missing). The token counts are the engine's own `PREFILL-DONE` numbers.

## 5. Limits of these tests (read before relying on a number)

- Single runs per K for F3/F5; only F2 has 7 order-balanced repetitions. Order was not randomized across K values.
- "min free" comes from 250 ms nvidia-smi samples and can miss a short allocation spike; the engine's own `cudaMemGetInfo` lines (`[VRAM-SLOTS]`, `VRAM left free`, `GGML-QWEN-FULL-VRAM-TRACE`) are phase snapshots, not a continuous minimum.
- The stress session covers two prompt sizes (28, 402), five requests, 48 generated tokens, context 2048. Untested: prompts of ~100-300 and 1,000+ tokens, dozens of requests per process, filling the 2,048 context, a different context size, long runs for fragmentation.
- Coverage is a survey prediction (survey recorded on other prompts); live hit rate was not measured.
- Output text was checked by eye (first ~60-120 characters); only F6 compares logits, and not yet at K >= 13,100.
- The floor override trades the allocator's safety margin for more experts; the only evidence that this is safe is the runs above.

## 6. Open items

Stress the 100-300 and 1,000+ token cases and a 15+ request session; teacher-forced check at the chosen K; decide the shipped K (14,458-14,554 passes the default floor and keeps 1.4-1.6 GiB free);
stage 2 of the policy (full graph default, hard errors instead of the five layered-decode fallback exits in `ggml_qwen4exp_graph.cpp`); stage 3 (delete layered leftovers for Qwen4Exp).
