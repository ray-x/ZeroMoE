#!/usr/bin/env bash
# Source this (". bench/glm_recipe_env.sh") before launching the engine for the 15 tok/s GLM-5.3-Flash dual-GPU recipe.
#   GGML_KERNEL_LIBS  directory holding libggml-cpu.so.0 with the AVX-512VL IQ2_XXS dot (ggml fork 36f489d6b or later);
#                     the engine's default linked ggml (built 2026-10-04) does not have it: 14.91 -> 15.43 tok/s.
#   GLM_RANKING       usage file used for both the 3090 cache (toml pin_file) and the 5060 Ti hot set.
export CUDA_VISIBLE_DEVICES=0,1
export MOESTREAM_GLM5NEXT_GPU1_HOT_GIB=${MOESTREAM_GLM5NEXT_GPU1_HOT_GIB:-15.0}
export MOESTREAM_GLM5NEXT_CPU_THREADS=${MOESTREAM_GLM5NEXT_CPU_THREADS:-10}
GGML_KERNEL_LIBS=${GGML_KERNEL_LIBS:-$HOME/work/ggml-iq2xxs/lib_new}
ORIG=$HOME/Projects/moestream-build-upstream-master/ggml-build/src
export LD_LIBRARY_PATH=$GGML_KERNEL_LIBS:$ORIG:$ORIG/ggml-cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
GLM_RANKING=${GLM_RANKING:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/docs/ssd-offload/data/glm53flash_usage_base_plus_decode100.txt}
export MOESTREAM_GLM5NEXT_OFFLOAD_USAGE=$GLM_RANKING MOESTREAM_GLM5NEXT_GPU1_HOT_USAGE=$GLM_RANKING
# The 5060 Ti holds only layers >= 31 (the 3090's pin_file ranking has those layers zeroed, pin_global_floor_per_layer = 0), so a
# layer's hot set is on one GPU: +3% on held-out prompts, neutral on the benchmark sequence.
export MOESTREAM_GLM5NEXT_GPU1_MIN_LAYER=${MOESTREAM_GLM5NEXT_GPU1_MIN_LAYER:-31}
# MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB unset = auto (available RAM - 36 GiB)
