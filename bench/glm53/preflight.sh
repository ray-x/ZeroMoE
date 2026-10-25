#!/bin/bash
# Refuse to start if another inference engine is running; then drop the page cache of the model/store/mirror files.
# Ignore this script's launcher ancestors: the gate's own command line can name the candidate
# binary (e.g. ./build_r2/zeromoe) even though it runs the private symlink. The Ling runner
# uses the same ancestor exclusion.
ancestors=" $$ "
p=$$
while :; do
  p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  [[ -z "$p" || "$p" -le 1 ]] && break
  ancestors+=" $p "
done
bad=$(ps -eo pid=,user=,cmd= | grep -E "zeromoe|llama-(server|cli|bench)|llama\.cpp|vllm|ollama|exllama|ggml" | grep -vE "grep|preflight|claude|codex|serena|headroom|cmake|make |nvcc|cc1plus|g\+\+|cicc|ptxas|fatbinary|cudafe" | while read -r pid user cmd; do
  case " $ancestors " in
    *" $pid "*) ;;
    *) echo "$pid $user $cmd" ;;
  esac
done )
if [ -n "$bad" ]; then echo "PREFLIGHT FAIL: other engine process(es) running:"; echo "$bad" | cut -c1-160; exit 1; fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then echo "PREFLIGHT FAIL: GPU in use:"; nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader; exit 1; fi
sync
python3 - <<'PY'
import glob
import os

files = []
for value in (os.environ.get("GLM_MODEL", ""), os.environ.get("GLM_STORE", ""),
              os.environ.get("GLM_MIRROR_GLOB", "")):
    if not value:
        continue
    if os.path.isdir(value):
        files.extend(glob.glob(os.path.join(value, "*.gguf")))
    else:
        files.extend(glob.glob(value))
n=0
for f in files:
    try:
        fd=os.open(f,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd); n+=1
    except OSError: pass
print("dropped page cache of",n,"files")
PY
rm -f /dev/shm/zeromoe* 2>/dev/null
echo "PREFLIGHT OK: $(grep MemAvailable /proc/meminfo | awk '{printf "%.1f GB", $2/1048576}') available, $(df -h /dev/shm /tmp 2>/dev/null | awk 'NR>1 && $3!="0" {printf "%s used on %s; ", $3, $6}')"
