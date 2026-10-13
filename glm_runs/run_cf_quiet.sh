#!/bin/bash
# Re-runs the confidence arms (k=1 K=1, k=2 K=2) once the box is quiet (no compile jobs, load < 2): a CUDA build from another session
# inflated the earlier attempts. Same async/lru settings as the other lookahead arms.
cd "$(dirname "$0")"
quiet() {
  until [ "$(awk '{print int($1)}' /proc/loadavg)" -lt 2 ] && ! pgrep -x 'cicc|ptxas|nvcc|cc1plus|cmake|ninja|make' >/dev/null; do
    sleep 20
  done
}
export MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC=1
quiet
env POLICY=lru PREFETCH=1 MOESTREAM_GLM5NEXT_PILOT_LOOKAHEAD=1 ./run_seg_ab.sh cf_k1_p1
quiet
env POLICY=lru PREFETCH=2 MOESTREAM_GLM5NEXT_PILOT_LOOKAHEAD=2 ./run_seg_ab.sh cf_k2_p2
echo CFDONE > pinab/cf.done
