#!/usr/bin/env bash
# GLM-5.3-Flash GSQ-RCO offload-pool run (the config behind the 14.1-14.2 tok/s dual-GPU baseline).
#   usage: bench/glm_gsqrco_pool.sh TAG [POOL_GIB=88] [prompt-file]     run from the repo root
#   env:   GLM_MODEL=/path/to/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf (required)   CVD=0,1 (default; 0 = 3090 only)   GPU1_HOT_GIB=14.2   CPUT (decode CPU threads, default 10)   PREFILL_THREADS=6
# Refuses to start unless the GPUs are idle, no other model run is active, the load average is under 3 and 100 GiB are free:
# a loaded box gives invalid numbers (0.89 vs 5.8 tok/s) and the 88 GiB pool leaves only ~16-30 GiB of headroom. A 1 Hz watchdog
# stops the engine below 6 GiB MemAvailable. Writes TAG.log, TAG.out and TAG.mem.csv (memory, load) next to TAG.
set -u
TAG=$(realpath -m "${1:?tag}"); POOL=${2:-88}; PROMPT=${3:-}
GLM_MODEL=${GLM_MODEL:?set GLM_MODEL to the GSQ-RCO .gguf}; CVD=${CVD:-0,1}; HOT=${GPU1_HOT_GIB:-14.2}; LOW_GIB=6
CFG=configs/bench/glm53flash_gsqrco_offload_pool.toml
RANK=${GLM_RANKING:-docs/ssd-offload/data/glm53flash_usage_combined_old_plus_survey5.txt}
[ -n "$PROMPT" ] || { PROMPT=$(mktemp); echo "Write a Python function that checks whether a string is a palindrome, and explain it." > "$PROMPT"; }
quiet() {
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
  pgrep -x "zeromoe|llama-bench|llama-cli|llama-server" > /dev/null && return 1
  awk '/MemAvailable/{exit !($2>100*1048576)}' /proc/meminfo || return 1
  awk '{exit !($1<3)}' /proc/loadavg
}
for _ in $(seq 1 1440); do quiet && { sleep 30; quiet && break; }; sleep 10; done
quiet || { echo "box never became quiet"; exit 2; }
sed -e "s#^log_path.*#log_path = \"$TAG.log\"#" -e "s#/path/to/models/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf#$GLM_MODEL#" -e "s/^graph_offload_ddr_gib = .*/graph_offload_ddr_gib = $POOL/" "$CFG" > "$TAG.toml"
{ cat "$PROMPT"; printf '\n:q\n'; } > "$TAG.in"
rm -f "$TAG.out" "$TAG.mem.csv"
ENVV="CUDA_VISIBLE_DEVICES=$CVD ZEROMOE_GLM5NEXT_ALLOC_FLOOR_MIB=512 ZEROMOE_GLM5NEXT_PREFILL_CPU_THREADS=${PREFILL_THREADS:-6}"
[ "$CVD" = "0" ] || ENVV="$ENVV ZEROMOE_GLM5NEXT_GPU1_HOT_GIB=$HOT"
[ -z "${CPUT:-}" ] || ENVV="$ENVV ZEROMOE_GLM5NEXT_CPU_THREADS=$CPUT"
nohup bash -c "$ENVV timeout 2400 scripts/run_capped.sh --limit-gb 118 -- ./build/zeromoe --config $TAG.toml --temp 0 --predict 100 \
  --force-start --graph-offload-usage-file $RANK < $TAG.in > $TAG.out 2>&1; echo rc=\$? >> $TAG.out" > /dev/null 2>&1 &
echo "t_s,avail_gib,used_gib,load" > "$TAG.mem.csv"; T0=$(date +%s)
while true; do
  av=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
  echo "$(( $(date +%s) - T0 )),$av,$(free -g | awk 'NR==2{print $3}'),$(cut -d' ' -f1 /proc/loadavg)" >> "$TAG.mem.csv"
  if awk -v a="$av" -v l="$LOW_GIB" 'BEGIN{exit !(a<l)}'; then
    echo "WATCHDOG: MemAvailable $av GiB < $LOW_GIB GiB - stopping the engine" | tee -a "$TAG.mem.csv"
    pkill -TERM -x zeromoe; sleep 3; pkill -KILL -x zeromoe; echo "rc=killed-by-watchdog" >> "$TAG.out"; break
  fi
  grep -q "^rc=" "$TAG.out" 2> /dev/null && break
  sleep 1
done
# steady decode = tokens 21..N of the engine's own per-token time
grep "FULL-DECODE-TIME" "$TAG.log" | sed 's/.*total=//; s/ ms//' | awk '{v[NR]=$1} END{for(i=21;i<=NR;i++){s+=v[i];n++} if(n) printf "steady %.1f ms/token = %.2f tok/s (n=%d)\n", s/n, 1000/(s/n), n}'
grep "GLM5NEXT-LANE\] tokens" "$TAG.log" | tail -1 | cut -c1-260
echo "min MemAvailable $(awk -F, 'NR>1{if(m==""||$2<m)m=$2}END{print m}' "$TAG.mem.csv") GiB, mean load $(awk -F, 'NR>1{s+=$4;n++}END{printf "%.1f", s/n}' "$TAG.mem.csv")"
