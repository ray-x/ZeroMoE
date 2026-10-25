# Ling 3.1 Flash integration investigation

## Inputs inspected

- Worktree: `ling-3.1-flash`, based on `master` at `c03d677`.
- Model: `/home/ray/vllm-workspace/ling-3.1-flash/Ling-3.1-flash-AD-IQ1_S-00001-of-00004.gguf` and its three companion shards.
- Reference implementation: `/home/ray/Projects/llama.cpp/src/models/bailingmoe3.cpp`.
- Existing ZeroMoE references: Qwen3.8 Flash Next (`qwen4exp`) and GLM5Next.

This records the integration and a full-checkpoint full-graph comparison
against llama.cpp. The parity run used the prompt `Hi there`, with matching
token IDs, and compared prefill plus the first decode step.

## Implementation

The live engine now detects `bailingmoe3`, validates its KDA/MLA layer map
and expert groups, allocates recurrent and MLA caches, preloads the family's
attention tensors, and dispatches its KDA and gated MLA attention. The MoE
router filters to the four selected expert groups before top eight selection.
The tokenizer recognizes the `bailingmoe2` pre-tokenizer's single-digit split,
and the interactive prompt path renders Bailing V3's system, human, and
assistant tags. `Ling31FlashGraphModel` enables the 72-layer GGML graph for
both prefill and decode when `ZEROMOE_LING_FULL_GRAPH=1`. It reuses the
GLM5Next graph lifecycle and router graph operations with Ling's own KDA,
RoPE-split MLA, and attention-gate equations. The offload aliases configure
layers 4–71 to read routed experts from the four GGUF shards, with an 8 GiB
dynamic expert cache; layers 0–3 use the declared dense FFN.

The full-checkpoint graph log advanced through layer 72/72 in both prefill
and decode. The decode graph reported 63 KDA layers and the 9 MLA layers.
The run generated two tokens, with no CUDA allocation failure or graph
fallback.

The changed C++ translation units compiled against headers from the local
llama.cpp GGML tree. Its prebuilt GGML libraries lacked ZeroMoE's stream and
captured-graph bridge symbols; an older ZeroMoE GGML build lacked precision
API used by existing GLM and DeepSeek code. The final local build used a
temporary copy of current GGML with the bridge exports and `ggml_graph_view`
visibility change. Patches 0011 and 0012 track those compatibility edits.

## Checkpoint facts

All four shard headers parse as GGUF v3 and declare 1,603 tensors in four splits. Their tensor tables cover layers 0–71. The architecture is `bailingmoe3`; the model name is `Ling-3.1-flash`.

| Property | GGUF value |
| --- | --- |
| Hidden size / layers / heads | 4096 / 72 / 64 |
| Attention schedule | 63 KDA layers; MLA at layers 7, 15, 23, 31, 39, 47, 55, 63, 71 |
| KDA | 128 head width; convolution width 4; safe gate; lower bound -5 |
| MLA | Q LoRA 1536; KV LoRA 512; key 192 including 64 RoPE dimensions; value 128 |
| MoE | 512 experts, top 8, groups 8/4, sigmoid gate, weight normalization, scale 2.5 |
| FFN | First 4 layers dense at width 10240; routed and shared experts at width 1280 |
| Vocabulary / context | 157184 / 262144 |
| Quant types found | F32, F16, Q8_0, IQ2_XXS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S |
| Tokenizer | GPT-2/BPE with `bailingmoe2` pre-tokenizer; BOS 156891, EOS 156895 |

The `attention.head_count_kv` array determines the layer type: zero selects KDA and one selects MLA. Both `swiglu_clamp_exp` and `swiglu_clamp_shexp` are 72 element arrays, mostly zero with some 10.0 entries. They must remain per layer.

## Full-checkpoint parity

The `IQ1_S` full checkpoint was evaluated in both llama.cpp and ZeroMoE.
Both used the raw prompt `Hi there`, tokenized as `[10754, 947]`. The ZeroMoE
prompt-token dump matches llama.cpp. Greedy continuation IDs were `11` and
`331` in both runtimes.

| Row | Phase | Max absolute delta | RMSE | Cosine similarity | Top-1 |
|---:|---|---:|---:|---:|---|
| 0 | Prefill | 0.3203249 | 0.06291843 | 0.999779617 | 11 / 11 |
| 1 | Decode | 1.255443 | 0.2285801 | 0.997259549 | 331 / 331 |

These are raw logits before sampling adjustments. The top token matches on both
rows; the nonzero logit deltas are retained here rather than described as
bit-exact parity. The comparison covers one short prompt and the first decode
step; it does not establish long-context or broad prompt parity.

The run configuration is [ling-3.1-flash-fullgraph.toml](../configs/ling-3.1-flash-fullgraph.toml).
It used GPU 0 (RTX 3090), a 1.5 GiB persistent expert cache, 16 active-layer
slots, host budget 32 GiB, and the 8 GiB full-graph offload cache. It was run
as a transient systemd service with `MemorySwapMax=0` and `MemoryMax=40G`; the
ZeroMoE process reported 20.78 GB peak RSS. The run log reached layer 72/72
in prefill and decode, and the decode graph reported 63 KDA plus 9 MLA layers.

## Build note

No Ling-specific CUDA kernel implementation was added. The graph uses existing
GGML CUDA operations and FATTN. The full-graph adapter needs stream and captured
graph handles from GGML; the local newer ggml source did not accept the older
bridge hunk in patch 0003, so the final local build used equivalent bridge
exports in a temporary ggml copy. Patches 0011 and 0012 now track the bridge
compatibility hunk and the `ggml_graph_view` shared-library export needed by
that build. A CUDA backend rebuild is required when those GGML changes are
applied; it does not recompile Ling-specific kernels because none were added.

## Remaining validation scope

- The tested raw prompt exercised one KDA/MLA path through the full 72-layer
  graph and the first decode step. More prompts and longer contexts remain to
  be measured.
- The `IQ1_S` and `IQ3_XXS` routed tensors are loaded through the graph/offload
  path; this comparison is evidence for the tested tokens, not a general
  guarantee for every tensor type or route.
