cd /home/ray/work/moestream-graph/qwen_runs
for t in n48_r1 n96_r1 n191_r1 n280_r1; do
  echo "== $t"
  grep -aiE "OPTION-B|resident" $t.log | head -5 | cut -c1-220
  head -c 200 $t.tokens; echo
done
grep -a -ciE "neural|gradient" n280_r1.stdout
