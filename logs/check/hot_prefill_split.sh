#!/bin/bash
cd /home/ray/work/moestream-glm53
MOESTREAM_GLM5NEXT_PREFILL_GPU1=0 HOT=11 ./logs/check/run_check.sh short_hot11_pfgpu1off graph logs/check/short.txt 12
MOESTREAM_GLM5NEXT_PREFILL_GPU1=0 HOT=11 ./logs/check/run_check.sh long_hot11_pfgpu1off graph logs/check/long.txt 16
