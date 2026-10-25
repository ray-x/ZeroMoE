# Residual-norm trace comparison: ZeroMoE vs colibri

This harness settles one open question: **does ZeroMoE's residual-stream
|hidden| norm still diverge from colibri's in current code, and if so, where
(which layer, which sub-block) does the divergence start and accelerate?**

colibri's author observed (hardcoded in `glm.c:1326-1329`) that ZeroMoE's
residual grew ~30× faster than colibri's by the last layers, starting within
~2× at layer 0. But all our existing trace logs are stale (Jul 18, pre several
documented fixes) and from incomparable captures. This produces a **fresh,
controlled** comparison so we know whether the bug still exists and where to
look.

## What it compares

Both engines, at full per-layer resolution, print the L2 norm of the residual
stream for the **last token of the prefill batch**, three times per layer:

```
[TRACE l=N] in        |hidden|=X   # residual entering the layer (pre-attention)
[TRACE l=N] post-attn |hidden|=X   # residual after the attention residual-add
[TRACE l=N] post-ffn  |hidden|=X   # residual after the FFN/MoE residual-add
```

`diff_trace.py` parses both logs with one regex, takes the **first** occurrence
of each (layer, stage) — i.e. the **prefill** pass, ignoring decode-step lines —
aligns them, and prints the ZeroMoE/colibri ratio per (layer, stage) plus the
layer indices where the ratio first crosses 2× / 5× / 10× / 30×.

## Files

| file | purpose |
|---|---|
| `run_zeromoe.sh` | Run ZeroMoE on a prompt at `--log-level VERBOSE`, trace to a log |
| `run_colibri.sh`   | Run colibri (CPU) on the same prompt; copies `/tmp/colibri_trace.log` out |
| `diff_trace.py`    | Parse + diff the two logs, print the per-layer ratio table |

## Prerequisites (set once)

1. **ZeroMoE built** on the Linux+CUDA box: `./build/zeromoe` exists.
   Know which config you normally run (e.g. `configs/glm52_dual.toml`).
2. **colibri built (CPU)**: `cd /path/to/colibri/c && make glm`. CPU-only is the
   default (`CUDA=0`); do NOT set `COLI_CUDA=1` on a CPU-only binary.
3. **colibri's int4 model**: HF `jlnsrk/GLM-5.2-colibri-int4` on disk, with
   `config.json` + `tokenizer.json` + the `.safetensors` shards in one dir.

Set the paths via env vars (defaults are placeholders you must override):

```bash
export ZEROMOE_CONFIG=configs/glm52_dual.toml
export ZEROMOE_BIN=./build/zeromoe
export COLIBRI_SNAP=/path/to/GLM-5.2-colibri-int4
export COLIBRI_GLM=/path/to/colibri/c/glm
```

## Run it

```bash
cd /path/to/ZeroMoE
PROMPT="Hello"   # short, pure-ASCII, ~1 BPE token in both tokenizers

bash scripts/trace_compare/run_zeromoe.sh "$PROMPT" ms_trace.log
bash scripts/trace_compare/run_colibri.sh   "$PROMPT" colibri_trace.log
python3 scripts/trace_compare/diff_trace.py ms_trace.log colibri_trace.log
```

That's it. Paste the `diff_trace.py` output back.

## How to read the result

**Look at the per-layer `post-ffn` ratio series** (the compounding curve printed
at the bottom) and the threshold-crossing table.

- **Healthy / bug-gone**: ratios wander in roughly **0.5×–2×** throughout, maybe
  drifting to ~3× at the last layers. That's within the noise floor (see
  caveats below). The 30× claim would be stale.
- **Bug still present**: ratios start near ~2× at layer 0 and **accelerate
  monotonically** through 5× → 10× → 30× by the last layers. The exact layer
  range where it accelerates localizes the bug:
  - diverges mostly across **post-attn → in** steps → attention path
    (scale / RoPE / absorption / accumulation dtype)
  - diverges mostly across **post-ffn → in** steps → MoE/FFN path
    (already audited as byte-identical to colibri — re-check only if this points here)
  - diverges at **l=0 `in`** itself (before any layer math) → embedding /
    tokenization seed

## Critical comparability caveats — read before over-interpreting

1. **The two engines use DIFFERENT weight quantizations of the same base model.**
   ZeroMoE = GGUF `IQ2_M` (experts `IQ2_XXS`/`IQ3_XXS`); colibri = its own
   per-row `int4` requant of the FP8 checkpoint. So residual norms will **never**
   be bit-identical even with perfectly correct math. Expect per-layer ratios to
   wander. A *sustained, accelerating* drift to 10×–30× is the signal; a flat
   1.5×–3× is not.

2. **Different tokenizers + chat templates.** ZeroMoE's REPL auto-applies the
   GLM chat template (`[gMASK]<sop><|user|>\n{msg}<|assistant|>\n`) and uses an
   approximated BPE pretokenizer; colibri's `glm` binary takes a **raw** prompt
   (no template) when called directly (its `coli` wrapper adds a *different*
   template). So the two prefills process **different token sequences**. This
   adds noise — which is why a single short ASCII prompt ("Hello") is used.
   - The **`l=0 in` row is the seed diagnostic**: it's the embedding of the first
     prompt token before any layer math. If ZeroMoE's `l=0 in` is already ~2×
     colibri's, the divergence is seeded at the embedding/tokenization, not the
     layer math (matching the author's "~2× at layer 0" note).
   - If you need to tighten this (e.g. the run is ambiguous at 3–5× and you
     can't tell noise from signal), the next step is matching the exact token
     sequence: feed colibri the templated prompt via its `coli` wrapper, or
     bypass ZeroMoE's template, so both engines embed the identical tokens.

3. **`diff_trace.py` takes the first (prefill) occurrence** of each (layer,
   stage), so the one decode step from `--ngen 1` is ignored. If you change
   `--ngen`, the script still works as long as prefill ran.

## If the bug is confirmed

The accelerating layer range + which stage (`in`/`post-attn`/`post-ffn`) tells
us where to dig. Given the MoE weighting is already verified byte-identical to
colibri, the likely culprits (in priority order) are: a compounding error in
attention (scale, RoPE, the absorbed `q_absorbed`/`WUK`/`WUV` matmuls, or
accumulation dtype), or in RMSNorm (`eps` / the mean-subtraction term). The
layer range from this diff tells us which.
