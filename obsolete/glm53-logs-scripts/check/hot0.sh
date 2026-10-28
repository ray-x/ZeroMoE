#!/bin/bash
cd /path/to/moestream-glm53
for t in short:12 long:16; do P=${t%%:*}; G=${t##*:}
  HOT=0 ./obsolete/glm53-logs-scripts/check/run_check.sh ${P}_graph_hot0 graph obsolete/glm53-logs-scripts/check/$P.txt $G
done
