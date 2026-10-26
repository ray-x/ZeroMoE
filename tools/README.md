ld

# tools/

`gguf_tool` — a standalone utility that ships alongside ZeroMoE but builds
independently of it. Pure file I/O — no CUDA, no ggml, no OpenMP — so it
builds and runs on any machine with a C++20/23 compiler, including one
without a GPU (this is how it was developed and tested: on a darwin/Apple
Silicon box that can't build the main `zeromoe` engine at all).

One binary, four subcommands:

| Subcommand                                                           | What it does                                                             |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [`pack`](#pack--align-or-unpack-a-gguf-model-for-o_direct-streaming) | Re-align (default) or unpack GGUF tensors for `O_DIRECT` streaming.      |
| [`verify`](#verify--pre-flight-completeness--byte-compare-gate)      | Pre-flight completeness gate + optional byte-compare between two models. |
| [`io`](#io--crossplatform-per-expert-read-pattern-ab-benchmark)      | A/B per-expert read-throughput benchmark, original vs repacked.          |
| [`plan`](#plan--dense-moe--system-sizing-advisor)                     | Read-only sizing advisor: dense/MoE VRAM requirement, detected GPU/RAM, `.zeromoe_usage` summary. |

> **Prerequisite — MoE model only.** These tools only apply to
> Mixture-of-Experts models whose GGUF files contain Unsloth-style packed
> expert tensors (`blk.N.ffn_{gate,up,down}_exps.weight`). Dense models
> (Gemma, Llama, Mistral base, etc.) have no routed experts and will fail
> with `no *_exps.weight tensors`. Suitable models include DeepSeek-V3/R1,
> GLM-5.2, Mixtral (Unsloth quants), and similar large MoE architectures.

## Build

```bash
# From the repo root, via the wrapper (rebuilds if stale, then runs):
./tools/run_gguf_tool.sh <pack|io|verify> [args...]

# Or directly:
c++ -std=c++23 -O2 -Iinclude tools/gguf_tool.cpp tools/gguf_pack.cpp \
    tools/gguf_io.cpp tools/gguf_verify.cpp tools/gguf_plan.cpp -o build/gguf_tool

# Or via CMake (standalone tools/CMakeLists.txt, or the main build's target):
cmake -S . -B build && cmake --build build --target gguf_tool
# binary lands at ./build/gguf_tool
```

## EXL3 safetensors

ExLlamav3 exports are directories of safetensors files containing groups such
as `<linear>.trellis`, `<linear>.suh`/`<linear>.su`, and
`<linear>.svh`/`<linear>.sv`. Validate the complete directory before using it:

```bash
cmake --build build --target exl3_tool
./build/exl3_tool verify /path/to/model-exl3/
```

The verifier checks shard uniqueness, safetensors byte ranges, trellis bitrate,
scale dimensions, optional bias dimensions, and the `mcg`/`mul1` codebook
marker. It returns nonzero for an incomplete or inconsistent EXL3 export.

## End-to-end workflow

The workflow is: **pack → verify → (A/B test with `io`)**.

### 1. Pack

**Single shard:**

```bash
./build/gguf_tool pack /path/to/model.gguf ./repacked/ \
    --align 4096 --verify-all
```

**Whole directory of shards** (one-at-a-time, minimal extra disk usage):

```bash
./build/gguf_tool pack /path/to/model-shards/ ./repacked/ \
    --align 4096 --verify-all
```

Add `-d` / `--delete-source` only after you're satisfied with the
verification output — it deletes each source shard immediately after its
own verification passes, which keeps peak extra disk usage to roughly one
shard's worth rather than the whole model twice. **Never combine `-d`
with `--verify-samples 0`** (refused unless `--verify-all` is set).

### 2. Verify

**Always run this on the output directory before pointing the engine at
it.** This is the gate that catches an incomplete repack — e.g. a shard
that silently failed to process, leaving a layer missing
`blk.N.attn_k_b`/`attn_v_b`. Without this check that failure mode isn't
caught anywhere else: the engine doesn't error on a missing attention
tensor, it zeroes that layer's attention and produces garbage output while
otherwise looking healthy (lower VRAM use, even faster inference).

```bash
./build/gguf_tool verify ./repacked/
# MODEL COMPLETE, or a list of the missing blk.<L>.<tensor> entries + non-zero exit
```

Also useful as an exhaustive cross-check between two models (the thing a
single-expert spot-check never was):

```bash
./build/gguf_tool verify ./repacked/ --against /path/to/model-original/
```

### 3. A/B test (optional)

```bash
# Via the wrapper (rebuilds gguf_tool if stale, then runs):
./tools/run_gguf_tool.sh io /path/to/model-original.gguf ./repacked/ --reads 5000

# Directly:
./build/gguf_tool io /path/to/model-original.gguf ./repacked/ --reads 5000 --seed 42 --direct

# Via Make:
make bench-io ARGS="/path/to/model-original.gguf ./repacked/ --reads 5000"
```

The tool prints OLD and NEW throughput (MB/s), the NEW/OLD speedup ratio,
and (with `--direct`) a third O_DIRECT/DMA line for the repacked shard.
**Use a large model (multi-GB)** — if the whole file fits in the OS disk
cache both paths show memory-speed numbers and the ratio is noise.

---

## `pack` — align or unpack a GGUF model for O_DIRECT streaming

Rewrites a GGUF model (one shard file, or a whole directory of shards) into
a new GGUF model. Every quantized byte is copied verbatim — no
requantization, no precision change, no value is reinterpreted. The output
is still a completely standard, spec-valid GGUF file loadable by any
standard GGUF loader (`llama.cpp`, LM Studio, Ollama, etc.).

### Two modes

**`--align-only` (the default)** — re-aligns every tensor's start offset
to `general.alignment` (4096 bytes by default) and copies every tensor
byte-for-byte, **including packed `*_exps` tensors, unchanged**. This is
the recommended, engine-safe mode:

- Keeps the packed `*_exps` format ZeroMoE's engine natively loads and
  whose VRAM expert fast-path (`populate_vram_shadow`/`packed_slab_geometry`)
  requires — **zero engine changes needed**.
- Makes every expert slice start at a block-aligned offset, the
  prerequisite for `O_DIRECT`/DMA reads — just set `direct_io=true` (or
  `--direct-io`) in the engine config to get DMA on the repacked model.

**`--unpack` (opt-in)** — the older per-expert behavior: splits each packed
`blk.N.ffn_{gate,up,down}_exps.weight` (shape `[a, b, n_experts]`) into
`n_experts` individual tensors (`blk.N.ffn_{gate,up,down}.E.weight`), laid
out per-layer in `gate → up → down` order, packed tightly (no alignment
padding between the three projections, only each expert _block_'s start is
aligned) so the engine's automatic contiguity detection can fetch a whole
expert with one coalesced `pread`.

> **Caveat:** the engine's VRAM expert cache fast-path is packed-format-only.
> A model produced with `--unpack` still loads and runs correctly, but every
> expert access goes through the slower per-expert cold path instead of the
> packed-slab fast path. Use `--align-only` (the default) unless you have a
> specific reason to need the legacy per-expert layout.

#### Automatic Interleaved Read Detection

ZeroMoE's engine **automatically detects** contiguous per-expert layouts
(as produced by `--unpack`) and optimizes the read strategy:

- **Contiguous experts detected**: single coalesced `pread` fetches all
  three projections (gate+up+down) in one syscall. Saturates NVMe queue
  depth and sequential bandwidth. **Typical: 1.5–3× faster** than scattered
  reads on NVMe hardware.
- **Scattered layout** (original Unsloth-packed GGUF, or `--align-only`
  output — both keep the packed `*_exps` tensor): three separate `pread`
  calls per expert miss (or one, via the packed VRAM-shadow fast path when
  applicable), each incurring seek latency. Fully supported.

The optimization requires no code changes or flags — it's automatic based
on detecting whether the three projections sit at contiguous byte offsets
on disk.

See `CLAUDE.md`'s "Offline GGUF Tool" section for the full design
rationale, including two real correctness bugs this tool's existence
surfaced and fixed in the shared loader (`include/gguf_loader.hpp`):
array-typed KV entries (e.g. the entire tokenizer vocabulary) being
silently discarded by the engine's own parser, and the loader hardcoding
32-byte alignment instead of reading `general.alignment` from the file.

### Usage

```
gguf_tool pack <source_shard_file_or_dir> <output_dir> [--align N] [--max-shard-gb N] [--verify-samples N] [--verify-all] [--unpack] [-d|--delete-source]
## sample
./build/gguf_tool pack source_folder target_folder --verify-all -d
```

The first argument may be **either a single shard file or a directory**:

- **Single file** → processes just that one shard.
- **Directory** → walks every `*.gguf` file directly inside it (sorted,
  not recursive), processing them one after another — equivalent to
  invoking the tool once per file by hand, but automatic.

The second argument is **always an output directory**, never a target
file — even in single-file mode a source shard can be split across
multiple output shards if it exceeds `--max-shard-gb`, and output files
are always named into a directory (`repacked-<source_stem>-partNNNNN-of-
NNNNN.gguf`, tagged with the source file's own basename so separate
invocations writing into the same output directory never collide).

Pack one shard of a model:

```bash
./build/gguf_tool pack /path/to/model/shard-00001-of-00005.gguf /path/to/output --verify-all -d
```

Pack an entire model directory:

```bash
./build/gguf_tool pack /path/to/model /path/to/output --verify-all -d
```

Flags:

| Flag                    | Default | Meaning                                                                                                                                                                        |
| ----------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--align N`             | `4096`  | Byte alignment for every tensor's start offset. Must be a power of two.                                                                                                        |
| `--max-shard-gb N`      | `40`    | Approx. max size per **output** shard produced from each source file.                                                                                                          |
| `--verify-samples N`    | `200`   | Random tensors to byte-compare source vs. output after writing (`0` = skip verification).                                                                                      |
| `--verify-all`          | off     | Byte-compare **every** tensor (exhaustive; overrides `--verify-samples`). Recommended before trusting a repack.                                                                |
| `--unpack`              | off     | Opt-in: unpack `*_exps` into per-expert tensors instead of the default align-only pass. See the VRAM-shadow caveat above.                                                      |
| `-d`, `--delete-source` | off     | **Destructive.** Delete each source shard immediately after _its own_ verification passes. Refuses to combine with `--verify-samples 0` unless `--verify-all` is set. No undo. |

### Why one file at a time

The tool deliberately processes (and, with `-d`, deletes) exactly one
source shard at a time rather than requiring the whole source model and
the whole output model to coexist on disk simultaneously. For a
~150–250GB model, that's the difference between needing ~2x the model's
size in free disk space and needing roughly one shard's worth (a
model's shards are typically 25–40GB each). Directory mode walks the
shards in this same one-at-a-time fashion internally — peak extra disk
usage during a full-model conversion is "one source shard + its output,"
not "the whole model, twice."

Verification is mandatory in spirit even though `--verify-samples 0` is
technically allowed: it reloads the just-written output with ZeroMoE's
own real GGUF loader and confirms tensor counts plus a byte-exact
comparison against the source for the sampled (or, with `--verify-all`,
every) tensor. Read its output before trusting a repacked file, and never
pair `-d` with `--verify-samples 0` unless `--verify-all` is set. Note
that `pack`'s verification is necessarily scoped to the shard(s) it just
processed — it can't detect a _different_ shard that silently failed
earlier. Run `gguf_tool verify` on the whole output directory afterward
for that directory-wide guarantee.

---

## `verify` — pre-flight completeness + byte-compare gate

The check that would have caught an incomplete repack before the engine
ever saw it: a shard not fully repacked/present (e.g. a missing
`blk.63.attn_k_b`/`attn_v_b`) isn't detected by `io` (only looks at expert
tensors) or by `pack`'s own per-shard verification (checks tensors it
processed, not the whole model directory). `verify` checks the whole
directory.

### What it checks

1. **Attention + norm completeness (hard-required).** For ordinary models,
   every per-layer tensor-name suffix under `attn`/`norm` must appear in
   _every_ `blk.<L>` from the model's min to max layer. Qwen4Exp is checked
   architecture-aware: GDN layers use `attn_qkv`/`attn_gate`/`ssm_norm`,
   while full-attention layers use separate `attn_q/k/v` and indexer norms.
   Any tensor missing from its required layer type is reported and the
   process exits non-zero.
2. **FFN expert reporting (informational).** Detects packed `*_exps` vs
   per-expert format and `n_experts` purely from tensor names (no
   architecture/KV parsing needed). A layer legitimately lacking experts
   (e.g. a leading dense layer) is noted, not failed.
3. **`--against <other_model>` (optional).** Exhaustive byte-compare of
   every tensor common to both models — sizes and full byte contents, not
   just a sampled expert.

### Usage

```
gguf_tool verify <model_file_or_dir> [--against <other_model>]
## sample
./build/gguf_tool verify ./repacked/
./build/gguf_tool verify ./repacked/ --against /path/to/model-original/
```

Exit code `0` = complete (and, with `--against`, byte-identical on every
common tensor); non-zero = incomplete or mismatched — safe to use as a
CI/script gate ahead of the engine.

---

## `io` — cross-platform per-expert read-pattern A/B benchmark

Quantifies the streaming-layout improvement from `pack`. Given an original
Unsloth-packed model and its repacked counterpart, it simulates the
decode-time expert-access pattern (random layer+expert "misses") and
measures aggregate disk-read throughput for each side:

- **OLD (original)** — always plain buffered `pread`s, no DMA, no threads,
  no io*uring — a fixed baseline. 3 scattered `pread`s per expert (slicing
  `ffn*{gate,up,down}\_exps` at the expert's offset) against the original
  packed layout.
- **NEW (repacked)** — 1 tight `pread` per expert against the repacked
  layout; with `--direct`, an additional pass re-reads the same expert
  slices through `O_DIRECT`/DMA (Linux only — the original can't go
  `O_DIRECT` since Unsloth's packed offsets aren't block-aligned).

Verification is built in: all picked experts are byte-compared between the
two models (not just expert #0), so a benchmark run also doubles as a spot
correctness check.

**Important:** this benchmark measures `pread` throughput in user-space,
not kernel-level I/O. True production performance gains appear only on
**Linux + NVMe** systems where `O_DIRECT` is active and models are large
enough for reads to hit real disk (not cache). On macOS or with
cached/small models, both paths show memory-speed numbers and the A/B
ratio is noise. See "Performance Characteristics" below for realistic
expectations.

### Usage

```
gguf_tool io <orig_model> <repacked_model> [--reads N] [--seed S] [--direct]
## sample
make bench-io ARGS="/path/to/model.gguf ./repacked --reads 5000"
# or: ./tools/run_gguf_tool.sh io /path/to/model.gguf ./repacked --reads 5000 --direct
```

Use a large real model so reads hit disk; on a small fixture everything is
cached and the ratio is noise.

### macOS notes

| Concern              | Detail                                                                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Page-cache bypass    | `io` calls `fcntl(fd, F_NOCACHE, 1)` automatically — no extra setup                                                                              |
| `O_DIRECT`           | Not available on macOS; `--direct` is a no-op there — the production engine's `O_DIRECT` benefit is Linux/NVMe-only and is not measured on macOS |
| Meaningful A/B ratio | The file must be large enough that reads actually hit disk, not RAM                                                                              |

### Performance Characteristics

On **Linux + ext4/XFS + NVMe** with `O_DIRECT` enabled:

| Configuration                                                                                | Expected Throughput              | Expert Read Latency    |
| -------------------------------------------------------------------------------------------- | -------------------------------- | ---------------------- |
| **Original packed GGUF**                                                                     | 0.6–0.8 GB/s (seek-bound)        | ~3–5 ms per miss       |
| **Repacked (align-only, packed VRAM-shadow fast path)** or **repacked+unpacked interleaved** | 1.7–3.0 GB/s (bandwidth-limited) | ~0.8–1.5 ms per miss   |
| **Speedup**                                                                                  | **2–4×** over original           | **3–5× lower latency** |

Speedup is maximized on:

- **NVMe** drives (especially modern PCIe 4.0+ with deep queue depth)
- **Large models** (>100 GB) where reads hit real disk, not cache
- **Linux** with proper `O_DIRECT` support (macOS `F_NOCACHE` is weaker)

On **small models** or **cached workloads**, the optimization is invisible
since reads are served from RAM (in which case the kernel's scheduler may
already coalesce the original scattered reads). The layout changes add
zero overhead and 100% spec compatibility, so `pack --align-only` is
always safe to apply.

---

## `plan` — dense / MoE / system sizing advisor

Different models need different `dense_gb`/`moe_cache_gb` — GLM-5.2's dense
tensor set is not GLM-4.7-flash's. Rather than guess-and-check with real
runs, `plan` reads the model's own tensor metadata (no architecture/KV
parsing needed, same name-pattern approach as `verify`'s expert-format
detection) and this machine's config to print:

- **Dense (persistent, always-VRAM-resident) tensor requirement**, broken
  down by the SAME categories a real run's `[VRAM-BREAKDOWN]` log uses
  (`attn_output`, `other_attention`, `shared_expert`, `embedding + LM head`,
  `leading_dense_ffn`, `norms`) — with a worst-case (nothing BF16-converted)
  and best-case (every `attn_output` layer BF16-converted) GB range, so
  these numbers agree with what the engine will actually report for the
  same model.
- **MoE routed-expert requirement** — one expert's slab size (gate+up+down,
  the exact formula `engine.hpp`'s `compute_expert_slab_bytes` uses), full
  everything-resident size, and a GB→slots quick-reference table so a
  candidate `moe_cache_gb` translates directly to a slot count without
  waiting for a real run's `[GPU] Expert VRAM cache: N slots` log line.
- **Detected system config** — GPU name + total VRAM per device (via
  `nvidia-smi`, best-effort — silently skipped, not an error, on a
  no-NVIDIA/macOS box) and total system RAM (`/proc/meminfo` on Linux,
  `sysctl` on macOS).
- **`.zeromoe_usage` summary**, if the file exists — how many
  `(layer,expert)` pairs have real routing history, and per-layer top-K
  traffic concentration (for K=8/11/16) so `pin_topk_per_layer` can be
  chosen from real data instead of a guess, even before a first real run
  finishes.

Deliberately **advisory only** — prints numbers, never writes a config
file. It also does NOT estimate KV cache size: that needs architecture
constants (`kv_lora`/`qk_rope` for MLA, or `head_dim`/`n_head_kv` for dense
attention) that live in GGUF metadata KVs, which this lightweight,
name-pattern-only tool doesn't parse (that's `graph_initializer.cpp`,
engine-side, intentionally not linked into this zero-dependency binary) —
check a real run's own `"[GRAPH] KV cache: ... GB"` log line for that.

### Usage

```
gguf_tool plan <model_file_or_dir> [--usage-file <path>]
## sample
./build/gguf_tool plan /path/to/model/
./build/gguf_tool plan /path/to/model/ --usage-file /path/to/.zeromoe_usage
```
