#!/bin/bash
cd /path/to/moestream-glm53
i=0
while IFS= read -r line; do
  rm -f logs/survey_dsa/b$i.trace
  PROMPT="$line" MOESTREAM_GLM5NEXT_ROUTE_TRACE=/path/to/moestream-glm53/logs/survey_dsa/b$i.trace \
  MOESTREAM_GLM5NEXT_PREFILL_GPU1=0 MOESTREAM_GLM5NEXT_GPU1_HOT_GIB=11 MOESTREAM_GLM5NEXT_PILOT_STAGE=8 MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC=1 \
  MOESTREAM_GLM5NEXT_PILOT_PREFETCH=8 MOESTREAM_GLM5NEXT_PILOT_COPY=0 MOESTREAM_GLM5NEXT_PILOT_LOOKAHEAD=2 MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB=54 \
  ./obsolete/glm53-logs-scripts/run_dsa_survey.sh > logs/survey_dsa/run$i.out 2>&1
  echo "prompt $i done rc=$? lines=$(wc -l < logs/survey_dsa/b$i.trace 2>/dev/null)" >> logs/survey_dsa/progress.txt
  i=$((i+1))
done < obsolete/glm53-logs-scripts/survey_prompts.txt
echo ALLDONE >> logs/survey_dsa/progress.txt
