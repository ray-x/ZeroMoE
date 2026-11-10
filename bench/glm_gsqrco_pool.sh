#!/usr/bin/env bash
# GLM-5.3-Flash GSQ-RCO offload-pool run (the config behind the 14.1-14.2 tok/s dual-GPU baseline).
#   usage: bench/glm_gsqrco_pool.sh TAG [POOL_GIB=88] [prompt-file]     run from the repo root
#   extra: GLM_PRELOAD (more LD_PRELOAD libraries, e.g. bench/hostreg_trace.c built as .so)  GLM_EXTRA_ARGS (more engine arguments, e.g. --raw-prompt)  MOE_CACHE_GB (3090 expert cache GB)  GLM_RANKING (usage file for every hot-set/seeding decision)  ZEROMOE_BIN (engine binary, default ./build/zeromoe)  NGEN (tokens per prompt, default 100)  GLM_EXTRA_TOML (lines added to the
#          root of the config)  GLM_EXTRA_ENV (VAR=val ... for the engine)  a prompt file may hold several lines (one prompt each)
#   env:   GLM_MODEL=/path/to/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf (required)   CVD=0,1 (default; 0 = 3090 only)
#          GPU1_HOT_GIB=9.0 for an 11 GB 2080 Ti (14.2 on a 16 GB 5060 Ti); written as the root TOML key
#          graph_gpu1_hot_gib, which is what actually enables the second GPU on current master
#          CPUT (CPU threads, default 10; GLM prefill uses min(CPUT, 6))
# Refuses to start unless the GPUs are idle, no other model run is active, the load average is under 3 and 100 GiB are free:
# a loaded box gives invalid numbers (0.89 vs 5.8 tok/s) and the 88 GiB pool leaves only ~16-30 GiB of headroom. A 1 Hz watchdog
# stops the engine below 6 GiB MemAvailable. Writes TAG.log, TAG.out and TAG.mem.csv (memory, load) next to TAG.
set -u
TAG=$(realpath -m "${1:?tag}"); POOL=${2:-88}; PROMPT=${3:-}
GLM_MODEL=${GLM_MODEL:?set GLM_MODEL to the GSQ-RCO .gguf}; CVD=${CVD:-0,1}; HOT=${GPU1_HOT_GIB:-14.2}; LOW_GIB=6
CFG=configs/bench/glm53flash_gsqrco_offload_pool.toml
BIN=${ZEROMOE_BIN:-./build/zeromoe}; NGEN=${NGEN:-100}
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
# GLM_RANKING replaces the usage ranking everywhere it is read: the 3090 hot set (pin_file), the 5060 Ti hot set and the pool
# seeding (--graph-offload-usage-file). The shipped ranking is used when it is not set.
# MOE_CACHE_GB: the 3090 expert cache (config default 12.4; the dense arena of ~9 GiB must still fit, decode peaks at ~92% VRAM)
[ -z "${MOE_CACHE_GB:-}" ] || sed -i "s/^moe_cache_gb = .*/moe_cache_gb = $MOE_CACHE_GB/" "$TAG.toml"
[ -z "${GLM_RANKING:-}" ] || sed -i "s#^pin_file = .*#pin_file = \"$GLM_RANKING\"#" "$TAG.toml"
# The GLM settings resolver reads root TOML keys, not the old
# ZEROMOE_GLM5NEXT_* environment aliases. Inject the GPU1 hot budget and the
# allocator floor here instead; a config that already carries a key wins.
python3 - "$TAG.toml" "${GLM_EXTRA_TOML:-}" "$CVD" "$HOT" <<'PY'
import re, sys
p, extra, cvd, hot = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
s = open(p).read()

def has_key(text, key):
    # Only a real root assignment counts; a commented example must not suppress it.
    return re.search(r'^\s*' + key + r'\s*=', text, re.M) is not None

root = ""
if not has_key(s, "graph_alloc_floor_mib") and not has_key(extra, "graph_alloc_floor_mib"):
    root += "graph_alloc_floor_mib = 512\n"
# The second GPU is activated by graph_gpu1_hot_gib > 0, not by a CUDA_VISIBLE_DEVICES entry.
if cvd != "0" and not has_key(s, "graph_gpu1_hot_gib") and not has_key(extra, "graph_gpu1_hot_gib"):
    root += f"graph_gpu1_hot_gib = {hot}\n"
if extra:
    root += extra + ("" if extra.endswith("\n") else "\n")
i = s.index("[[gpus]]")
open(p, "w").write(s[:i] + root + "\n" + s[i:])
PY
{ cat "$PROMPT"; printf '\n:q\n'; } > "$TAG.in"
rm -f "$TAG.out" "$TAG.mem.csv"
gcc -shared -fPIC -O1 "$(dirname "$0")/abort_backtrace.c" -o "$TAG.abort_bt.so" 2> /dev/null || true
ENVV="LD_PRELOAD=$TAG.abort_bt.so${GLM_PRELOAD:+:$GLM_PRELOAD} CUDA_LOG_FILE=$TAG.cudalog CUDA_VISIBLE_DEVICES=$CVD"
[ -z "${GLM_EXTRA_ENV:-}" ] || ENVV="$ENVV $GLM_EXTRA_ENV"
nohup bash -c "$ENVV timeout 2400 scripts/run_capped.sh --limit-gb 118 -- $BIN --config $TAG.toml --threads ${CPUT:-10} --temp 0 --predict $NGEN ${GLM_EXTRA_ARGS:-} \
  --force-start --graph-offload-usage-file $RANK < $TAG.in > $TAG.out 2>&1; echo rc=\$? >> $TAG.out" > /dev/null 2>&1 &
echo "t_s,avail_gib,used_gib,load,other_cpu_hogs" > "$TAG.mem.csv"; T0=$(date +%s)
while true; do
  av=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
  echo "$(( $(date +%s) - T0 )),$av,$(free -g | awk 'NR==2{print $3}'),$(cut -d' ' -f1 /proc/loadavg),$(ps -eo pcpu,comm --sort=-pcpu | awk 'NR>1 && $2!="zeromoe" && $1>=20 {printf "%s:%s ", $2, $1}')" >> "$TAG.mem.csv"
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
echo "other processes >=20% CPU during the run: $(awk -F, 'NR>1 && $5!=""{print $5}' "$TAG.mem.csv" | tr ' ' '\n' | cut -d: -f1 | sort | uniq -c | sort -rn | head -5 | tr '\n' ' ')"
