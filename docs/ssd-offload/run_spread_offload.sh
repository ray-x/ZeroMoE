#!/bin/bash
# usage: run_off.sh NAME CONFIG   (OFFLOAD_DDR_GIB, THREADS optional)
cd /home/ray/work/moestream-graph/glm_runs
python3 - <<'PY'
import os,glob
for f in glob.glob('/home/ray/vllm-workspace/GLM-5.3-Flash/GLM-5.3-Flash-AJ-IQ2_XXS-*.gguf')+['/data/moestream-offload/glm53flash-aj-iq2xxs-cold3173.gguf']:
    fd=os.open(f,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
PY
export MOESTREAM_GLM5NEXT_OFFLOAD_FIRST_LAYER=3 MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB=${OFFLOAD_DDR_GIB:-32}
export MOESTREAM_GLM5NEXT_OFFLOAD_STORE=/data/moestream-offload/glm53flash-aj-iq2xxs-cold3173.gguf
export MOESTREAM_GLM5NEXT_OFFLOAD_MODEL=/home/ray/vllm-workspace/GLM-5.3-Flash/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf
export MOESTREAM_GLM5NEXT_OFFLOAD_USAGE=/home/ray/Projects/MoEStream/docs/ssd-offload/data/glm53flash_usage_base_plus_decode100.txt
export MOESTREAM_GLM5NEXT_OFFLOAD_THP=${OFFLOAD_THP:-0}
exec ./run_exp2.sh "$@"
