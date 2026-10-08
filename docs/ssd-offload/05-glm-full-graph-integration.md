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

### Prefetch estimate (decided not to build)

Replaying the 716 decode tokens of the survey trace through the pinned+LRU cache at 3,900 and 5,245 experts, a predictor that
prefetches "last token's (and the token before's) picks for this layer" covers 0-1% of the misses: consecutive tokens do not
repeat their cold experts. Only a pilot-router predictor (next layer's router applied to this layer's hidden state, measured
51-58% recall earlier in the engine) could help, and at 32 GiB (7 misses/token, ~21 ms) it would hide at most about half of
that for new graph nodes plus a thread-safe prefetch path around the cache and arena. The lever that works is expert DDR
(48 GiB: 0 misses, ~10 tok/s).

### Strict full-graph mode

`MOESTREAM_GLM5NEXT_GRAPH_STRICT=1` turns a declined decode graph into a hard error ("failing the request") instead of the
silent fallback to the layered path. Checked both ways: the best configuration runs normally under it (97.1 ms/token over 39
tokens), and an oversized GPU1 hot set (15.5 GiB, which the graph declines) aborts with the error. It is opt-in because
prefill still uses the layered path; making it the default needs a graph-only prefill first.

### Independent check against llama.cpp (neurall's llama-server, same GGUF, `--n-cpu-moe 99`)

Teacher-forced on the same 96 tokens with the engine's own prompt ids (`MOESTREAM_DUMP_PROMPT_TOKENS`), comparing
log-softmax of MoEStream's raw logits with llama.cpp's top-40 log-probs (`glm_runs/llama_ref.py`, `cmp_llama.py`):

| MoEStream configuration | steps where top-1 == llama.cpp | median mean-abs logprob delta (llama's top 10) |
|---|---|---|
| original whole-layer baseline | 74/96 | 1.222 |
| spread hot experts, no offload | 79/96 | 1.222 |
| spread + DDR/SSD offload + GPU1 hot (best config) | 74/96 | 1.208 |

The offload and GPU1 branch do not move the engine further from the independent reference; the best configuration sits at the
same distance as the original baseline, and at step 0 the spread variants pick `#` like llama.cpp while the whole-layer baseline
picks `A`. The ~1.2 logprob gap to llama.cpp is common to all of them (a MoEStream-vs-llama.cpp difference, not introduced here).

**Tokenizer finding.** For the test prompt the engine tokenizes "mixture-of-experts" as `' mixture' '-' 'of' '-' 'exp' 'erts'`
(20933 12 1055 12 4580 15534) whereas llama.cpp produces `' mixture' '-of' '-exp' 'erts'` (20933 8667 18347 15534). The engine's
BPE misses merges, so the two engines never saw identical prompts before this comparison. Separate bug, not addressed here.

**Update after the tokenizer fix** (engine pretokenizer now implements the glm4 split; 9/9 test strings, including CJK and
mixed whitespace, tokenize identically to llama.cpp, so the 24 prompt ids are the same in both engines). Teacher-forced on the
fixed whole-layer baseline's 96 tokens, `glm_runs/v2/`:

| MoEStream configuration | steps where top-1 == llama.cpp | median mean-abs logprob delta |
|---|---|---|
| original whole-layer baseline | 87/96 | 1.220 |
| spread + DDR/SSD offload + GPU1 hot (best config, 9.57 tok/s on this prompt) | 84/96 | 1.250 |

The best configuration stays within 0.03 logprob of the baseline's distance to llama.cpp (3 fewer top-1 matches out of 96,
on a model whose top-2 margins are often below 0.1). The remaining ~1.2 gap to llama.cpp exists in the untouched baseline
too, so it is an engine-vs-engine difference outside this work; its cause is not diagnosed.

### N+1 (pilot) prediction accuracy on GLM decode

Engine pilot router, legacy decode of 96 tokens (4,032 layer checks, `spread_base.toml`, `FULL_GRAPH=0`), recall measured against the
real router's top-8 of the next layer:

| pilot_k (experts predicted per layer) | real top-8 slots covered | correct experts per prediction | by real rank (0 / 1 / 2 / 3 / 4 / 5 / 6 / 7) |
|---|---|---|---|
| 4 | 27.4% (2.19 of 8) | 2.19 of 4 = 54.8% precision | 73.2 / 60.6 / 47.8 / 37.4 / - |
| 8 | 39.5% (3.16 of 8) | 3.16 of 8 = 39.5% precision | 73.5 / 60.5 / 47.7 / 36.9 / 32.1 / 24.1 / 21.9 / 18.9 |

The predicted set is not restricted to cold experts, so some of the correct ones are already VRAM-hot; ranking only the non-hot experts
would raise the useful hit rate but was not measured.

#### Is the low accuracy a lead-time (input freshness) problem?

No. The DEPTH-1 pilot ranks layer l+1's experts from the residual stream after layer l's FFN (lead ~ one attention block); the ATTN-PILOT
variant uses the stream right after layer l's attention (lead ~ FFN(l) + attention(l+1), about 7x longer). Same 96 decode tokens, k=8:

| input | real top-8 slots covered | by real rank 0 / 1 / 2 / 3 |
|---|---|---|
| after layer l FFN (DEPTH-1) | 39.5% | 73.5 / 60.5 / 47.7 / 36.9 |
| after layer l attention (ATTN-PILOT) | 37.4% | 71.5 / 58.1 / 44.1 / 34.9 |

Two points of accuracy buy ~7x the lead, so the earlier input is the right one for prefetch. The ceiling is reusing the next layer's router on this layer's
residual stream. History-only predictors are much worse (`bench/next_layer_predictor_eval.py`, leave-one-prompt-out on the 5-prompt trace):
co-occurrence from the current layer's routed set gives precision@4 = 20.1% (15.6% on cold experts), a static usage prior 5.4%
(random is 2.8%).

### Corrected: next-layer prediction from exact hidden states (probe dump, `MOESTREAM_PROBE_DUMP`)

The engine's own pilot statistic (above, ~55% precision@4) turns out to be far below what the model's routing allows. Dumping the real
post-attention residual of each layer and the real router scores of every layer (6 prompts, ~3.2k tokens per layer; code/docs prompts,
layered prefill) and applying layer l+1's own norm+router to layer l's post-attention residual (the same input the ATTN-PILOT uses),
precision@k against the real top-8 of layer l+1 (`bench/next_layer_probe_fit.py`, leave-one-prompt-out, 41 layer pairs):

| method | k=1 | k=2 | k=3 | k=4 | k=5 | k=8 |
|---|---|---|---|---|---|---|
| pilot (next router on this layer's residual) | 97.0 | 93.8 | **89.7** | 85.0 | 80.1 | 65.6 |
| ridge probe alone (4096 -> 288) | 89.3 | 82.2 | 75.0 | 68.4 | 62.5 | 49.1 |
| pilot + ridge correction | 97.9 | 95.2 | 91.4 | 86.8 | 81.7 | 66.5 |

Worst layer for the plain pilot at k=3: layer 24 at 72.0% (80.2% with the ridge correction). On decode rows only (one prompt, 99 tokens x 41
layers, `bench/next_layer_pilot_decode_check.py`): k=1 96.2, k=2 92.9, k=3 88.8, k=4 83.9, k=5 78.8, k=8 64.6 (prefill rows of the same
run: 98.2 / 96.0 / 93.1 / 89.2 / 84.8 / 70.2). The router reimplementation matches the engine's dumped scores to 9.5e-4 (max abs diff).

So with exact hidden states the plain next-router reuse already reaches ~89% top-3 / ~84% top-4 / ~79% top-5, well above a 70% bar, and a learned probe adds
only ~2 points on average (it helps the worst layers). The engine's built-in pilot recall (37-40% at k=8 vs 65-70% here) is therefore an
engine-side measurement or input problem, not a property of the model; its cause is not yet identified.

### Why the engine's pilot read low: a stale ATTN-PILOT snapshot (fixed)

The engine's ATTN-PILOT copied the post-attention residual into `pilot_attn_snapshot` as soon as attention returned, before that residual was final in
`hidden_states`; the prediction therefore ran on a stale vector. Taking the snapshot immediately before the FFN (where the probe hook reads it) moves the
engine's own statistic on the same 96-token generation from **37.2% -> 70.6%** (rank 0: 71.6% -> 98.9%), matching the offline numbers (decode-row precision@8
71.0%, top-1 98.9%). Verified first that the hook's top-8 equals the engine's actually routed ids (13,104/13,104). The DEPTH-1 pilot (post-FFN state) stays at
39% because it predicts from a state that is further from layer l+1's router input than the post-attention one; the post-attention input is the one to use.
The fix is in `src/execution_layer.cpp` (snapshot re-taken right before the FFN). Hook-side numbers (overlap of 8) were 4-7 per layer vs 1-6 for the engine's
old snapshot on the same layers and token.

### Qwen3.8-Flash-Next (Q2_0): pilot and pilot + ridge

Probe taps in the Qwen full graph (`MOESTREAM_QWEN4EXP_PROBE_DUMP`: FFN input of every layer and its router probabilities), 6 prompts (~580-890 tokens, 64 decode
tokens each) on the accepted early-ready profile; pilot = layer l+1's router applied to layer l's FFN input, precision vs the real top-10 of layer l+1,
leave-one-prompt-out, 47 layer pairs (`bench/next_layer_probe_fit_qwen.py`):

| method | k=1 | k=2 | k=3 | k=4 | k=5 | k=10 |
|---|---|---|---|---|---|---|
| pilot | 90.9 | 88.5 | 85.8 | 83.1 | 80.3 | 64.5 |
| ridge probe alone | 89.1 | 85.7 | 82.5 | 79.4 | 76.2 | 61.3 |
| **pilot + ridge** | **94.8** | **93.0** | **91.2** | **89.1** | **86.8** | **72.0** |

Decode rows only (pilot): k=1 91.9, k=3 87.8, k=5 82.6, k=10 67.3. The two weakest layers for the plain pilot, 38 (44.4% at k=3) and 22 (58.6%), reach 71.6% and
72.2% with the ridge correction. So Qwen's next-layer routing is not hard to predict either: the plain pilot already gives ~86% top-3, and pilot + ridge ~91%.

### GLM cold-only budget (best configuration)

In the 48 GiB / GPU1-hot configuration 43.7% of routed experts are cold (17,131 of 39,168 over 120 tokens, i.e. ~3.5 cold experts per layer per token, each
~0.45 ms on the CPU). Restricting the pilot to non-hot experts (hot set approximated from the usage ranking at the same budget), staging the top-k:
top-2 82.1% precision / 37% of cold routes, top-3 73.9% / 50%, top-4 66.1% / 60%, top-6 52.8% / 72%.
