#!/bin/bash
cd /home/ray/work/moestream-glm53
for t in short:12 long:16; do P=${t%%:*}; G=${t##*:}
  for m in graph layered; do ./logs/check/run_check.sh ${P}_$m $m logs/check/$P.txt $G; done
done
