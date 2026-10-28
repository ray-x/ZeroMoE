#!/usr/bin/env bash
# Source this recipe, then pass "${GLM_GRAPH_ARGS[@]}" to the engine for GLM graph settings.
#   GGML_KERNEL_LIBS  directory holding libggml-cpu.so.0 with the AVX-512VL IQ2_XXS dot (ggml fork 36f489d6b or later);
#                     the engine's default linked ggml (built 2026-10-04) does not have it: 14.91 -> 15.43 tok/s.
#                     Required; GGML_BASE_LIBS optionally points at the matching ggml build's src directory.
#   GLM_RANKING       usage file used for both the 3090 cache (toml pin_file) and the 5060 Ti hot set.
export CUDA_VISIBLE_DEVICES=0,1
export ZEROMOE_GLM5NEXT_GPU1_HOT_GIB=${ZEROMOE_GLM5NEXT_GPU1_HOT_GIB:-15.0}
export ZEROMOE_GLM5NEXT_CPU_THREADS=${ZEROMOE_GLM5NEXT_CPU_THREADS:-10}
GGML_KERNEL_LIBS=${GGML_KERNEL_LIBS:?set GGML_KERNEL_LIBS to the directory containing the required libggml-cpu.so.0}
ORIG=${GGML_BASE_LIBS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/build/ggml-build/src}
export LD_LIBRARY_PATH=$GGML_KERNEL_LIBS:$ORIG:$ORIG/ggml-cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
GLM_RANKING=${GLM_RANKING:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/docs/ssd-offload/data/glm53flash_usage_base_plus_decode100.txt}
export ZEROMOE_GLM5NEXT_GPU1_HOT_USAGE=$GLM_RANKING
GLM_GRAPH_ARGS=(--graph-offload-usage-file "$GLM_RANKING")
# The 5060 Ti holds only layers >= 31 (the 3090's pin_file ranking has those layers zeroed, pin_global_floor_per_layer = 0), so a
# layer's hot set is on one GPU: +3% on held-out prompts, neutral on the benchmark sequence.
export ZEROMOE_GLM5NEXT_GPU1_MIN_LAYER=${ZEROMOE_GLM5NEXT_GPU1_MIN_LAYER:-31}
# graph_offload_ddr_gib unset uses auto sizing (available RAM - 36 GiB)

# Batched prefill on the GPU (805 tokens: 20 -> ~100 tok/s): cold experts are copied to VRAM per layer and read ahead by a
# background thread. Needs the ggml-cuda carrying patches 0006/0007 (bench/build_native.sh) and a >= 58 GiB DDR pool.
NATIVE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/build_native/ggml-build/src
if [ -f "$NATIVE/ggml-cuda/libggml-cuda.so.0" ]; then
  export LD_LIBRARY_PATH=$GGML_KERNEL_LIBS:$NATIVE:$NATIVE/ggml-cuda
  export ZEROMOE_GLM5NEXT_FULL_GRAPH_PREFILL=1
GLM_GRAPH_ARGS+=(--graph-prefill-op-offload 1)
  # --graph-offload-pinned-fraction 1.0 pins the whole DDR pool (~63 GiB of unreclaimable host RAM): prefill +0.3 s
  # faster (the prefetch ring copies straight from it), but it is the setting that overcommits the box when another large
  # job runs (two hard reboots on 2026-10-15). Default stays at the engine's 0.5; export 1.0 only when nothing else runs.
  :
fi
