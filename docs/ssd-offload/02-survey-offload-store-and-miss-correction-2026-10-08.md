# GLM-5.3-Flash: 5-prompt survey, the offload store, and a correction to the miss estimate (2026-10-08)

## 1. What was built
`gguf_tool offload` (tools/gguf_offload.cpp) ranks routed experts by traffic from a usage file, takes the coldest N
(ties broken round-robin across layers), and writes them to a single GGUF "offload store":
- each expert's gate, up and down planes are written back to back with no gap and the gate plane starts on a
  4096-byte boundary, so ONE aligned O_DIRECT read of [gate start, down end) fetches a whole expert;
- the file ends padded to 4096; `general.alignment` = 4096; tensors are named `offload.blk.<L>.ffn_<k>.<E>.weight`
  (deliberately not the unpacked-model names, because the engine requires an unpacked layer to be complete and a
  partial store must never be merged into the model by a directory scan); the (layer, expert) index is in the
  KV arrays `zeromoe.offload.layer/expert`. Pure byte copy, no requantization.
- `gguf_tool offload --verify <store> --against <model>` reads every expert back with O_DIRECT and byte-compares it
  with the source (also checks alignment and tightness).

Store for this model: `/data/moestream-offload/glm53flash-aj-iq2xxs-cold3173.gguf` (XFS on the second NVMe, KIOXIA).
- 3,173 experts, 9,519 tensors, 19.35 GiB (20,773,994,496 B, a multiple of 4096). Written in 18.5 s.
- verify: 3,173/3,173 experts byte-identical via O_DIRECT (20.77 GB); page-cache residency stays at 1 MB.
- fio, random whole-slab O_DIRECT reads (6,488,064 B): QD1 avg 2.23 ms, p99 2.44 ms, 2.9 GB/s; 4 jobs saturate at
  about 3.1 GB/s (8.2 ms per slab each).
- It is a 19 GiB file, not a few GiB: a GLM-5.3-Flash expert is 6.49 MB (three 2.16 MB planes), 3,173 x 6.49 MB.
- The main model file is unchanged (the store duplicates these experts); the engine will later skip them when
  filling DDR.

## 2. The survey
Five prompts through the existing `tests/moe_usage_survey/run_survey.py` in persistent mode (one engine process,
layered decode path, `--collect-usage-stats`, 160 new tokens each): 001 python/csv_average, 033 c_cpp/review_delete_vs_
delete_array, 065 os/deadlock, 081 daily/world_history, 105 general/summarize_distributed_systems. All five finished
with 0 error lines and coherent output. Config: tests/moe_usage_survey/survey_config_glm53flash.toml.
Result: 334,992 decode routing hits over all 42 MoE layers (271,184 on CPU layers 11-44) and 94,416 prefill hits.
Data: docs/ssd-offload/data/ (survey decode/prefill usage, and old+survey combined, which the store was built from).

## 3. Correction: static cold sets generalize badly
The estimate in doc 01 (about 1.5% route miss at 19 GiB on SSD) came from a RANDOM split of hits of one survey.
Hits inside one prompt are strongly correlated, so that split leaked the test prompts into the ranking. With
independent prompts the result is very different. Rank the coldest N experts on one survey, count the routes that land
on them in the other (CPU layers 11-44):

| N on SSD | GiB | miss, rank OLD -> test NEW | misses/token | MB/token | miss, rank NEW -> test OLD |
|---|---|---|---|---|---|
| 1,000 | 6.0 | 7.65% | 20.8 | 135 | 2.73% |
| 2,000 | 12.1 | 15.04% | 40.9 | 265 | 7.77% |
| 3,173 | 19.2 | 24.54% | 66.7 | 433 | 15.51% |
| 4,000 | 24.2 | 32.39% | 88.1 | 572 | 21.59% |

(old = the 2026-09-08 survey, composition not recorded; new = these 5 prompts; 272 routes per token on 34 layers.)
At 15-25% misses a static set costs 40-67 reads of 6.5 MB per token: 270-430 MB/token, i.e. about 80-120 ms/token if
exposed at 3.5 GB/s, against a 138 ms decode. A static set chosen offline from a handful of prompts does not work.
Only 256 experts of the 9,792 are unseen in both surveys, and the sets chosen by the two surveys overlap by only 1,929
(old vs combined) and 2,620 (new vs combined) of 3,173.

## 4. Consequences
- Expert use is prompt/domain specific. The design needs a DYNAMIC DDR cache over the SSD store (promote on miss,
  evict cold) so that a session's working set becomes resident; the static "offload set" is only the starting point.
- Misses then are mostly compulsory (first use of an expert in the session) plus capacity misses. Estimating them
  needs per-token route SEQUENCES, which the usage histograms do not contain.
- Next measurement: re-run the 5-prompt survey with router tracing (log_level VERBOSE, trace_filter TRACE-ROUTER) to
  capture (token, layer, expert ids), then simulate static+LRU/LFU/ARC caches at 40-56 GiB for single prompts and for
  the 5-prompt session, and a prefetch lookahead. Only then fix the policy and the staging-ring size.
- The tooling (store format, aligned slabs, verify) stays valid under either policy.

## 5. Caveats
- 5 prompts x 160 tokens is a small survey; the old survey's prompts are unknown.
- Read latencies measured with Codex occasionally compiling in the root tree.
