#!/bin/bash
cd /home/ray/work/moestream-glm53
for t in short:12 long:16; do P=${t%%:*}; G=${t##*:}
  HOT=0 ./logs/check/run_check.sh ${P}_graph_hot0 graph logs/check/$P.txt $G
done
