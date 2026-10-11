#!/bin/bash
# usage: [same env as run_seg_ab.sh] run_clean.sh NAME   -- waits for a quiet box, runs the arm, and retries (up to 3 times) when a compile
# job or another engine ran next to it. A contaminated attempt is kept as NAME.badK.* for the record and never used for timing.
D=$(cd "$(dirname "$0")" && pwd); N=$1
quiet() {
  until [ "$(pgrep -c -x 'cicc|ptxas|cc1plus|cc1|nvcc|cmake|ninja|make' 2>/dev/null || echo 0)" = 0 ] \
        && [ "$(awk '{print int($1)}' /proc/loadavg)" -lt 2 ] \
        && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] \
        && ! pgrep -x gguf_tool >/dev/null && [ ! -e "$HOME/work/.no-arms" ]; do sleep 20; done
}
for k in 1 2 3; do
  quiet
  rm -f $D/pinab/$N.done
  "$D/run_seg_ab.sh" "$N"
  bad=$(awk '$3>0 || $4>0' $D/pinab/$N.load.csv 2>/dev/null | wc -l)
  if [ "$bad" = 0 ]; then echo "$N clean (attempt $k)" >> $D/pinab/clean_runs.log; exit 0; fi
  echo "$N contaminated in $bad samples (attempt $k)" >> $D/pinab/clean_runs.log
  for f in $D/pinab/$N.*; do mv "$f" "${f/$N./$N.bad$k.}"; done
done
echo "$N gave up after 3 contaminated attempts" >> $D/pinab/clean_runs.log
exit 1
