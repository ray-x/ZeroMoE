#!/bin/bash
cd /home/ray/work/moestream-graph/glm_runs
mkdir -p pinsmaps
for pin in 0 1; do
  systemd-run --user --scope -q -p MemoryMax=100G -p MemorySwapMax=0 \
    env ZEROMOE_GLM5NEXT_OFFLOAD_PIN=$pin ZEROMOE_GLM5NEXT_PILOT_STAGE=3 CVD=0,1 ZEROMOE_GLM5NEXT_GPU1_HOT_GIB=14.2 OFFLOAD_THP=1 OFFLOAD_DDR_GIB=16 NGEN=60 ./run_off3.sh pinsm$pin $PWD/spread_off.toml &
  P=$!
  for t in $(seq 1 70); do sleep 5; E=$(pgrep -x zeromoe | head -1)
    if [ -n "$E" ] && [ -r /proc/$E/smaps_rollup ]; then
      cp /proc/meminfo pinsmaps/meminfo$pin.last; cp /proc/$E/smaps pinsmaps/smaps$pin.last 2>/dev/null; cp /proc/$E/smaps_rollup pinsmaps/rollup$pin.last
      echo "pin=$pin t=$((t*5)) rss_gb=$(( $(awk '/VmRSS/{print $2}' /proc/$E/status) / 1048576 )) $(free -g | awk 'NR==2{print "used="$3" cache="$6" avail="$7}')" >> pinsmaps/log.txt
    fi
    ps -p $P >/dev/null || break; done
  wait $P
done
echo SMAPSDONE >> pinsmaps/log.txt
