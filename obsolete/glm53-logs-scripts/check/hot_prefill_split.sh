#!/bin/bash
cd /home/ray/work/moestream-glm53
MOESTREAM_GLM5NEXT_PREFILL_GPU1=0 HOT=11 ./obsolete/glm53-logs-scripts/check/run_check.sh short_hot11_pfgpu1off graph obsolete/glm53-logs-scripts/check/short.txt 12
MOESTREAM_GLM5NEXT_PREFILL_GPU1=0 HOT=11 ./obsolete/glm53-logs-scripts/check/run_check.sh long_hot11_pfgpu1off graph obsolete/glm53-logs-scripts/check/long.txt 16
