#!/usr/bin/env bash
set -euo pipefail
worktree=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
profile=${LING_USAGE_FILE:-$worktree/bench/ling/reference-audit/baseline.pin}
log=${LING_LOG:-/tmp/ling-reference-$(date -u +%Y%m%dT%H%M%S).log}
config=$(mktemp /tmp/ling-reference-config.XXXXXX.toml)
python3 - "$worktree/bench/ling/reference-audit/config.toml" "$config" "$profile" <<'CONFIG'
from pathlib import Path
import json, os, re, sys
pool=int(os.environ.get('LING_DDR_GIB','72')); cap=int(os.environ.get('LING_CAP_GIB','110'))
assert pool in (64,68,72) and 0 < cap <= 110, 'Unsupported memory budget'
if os.environ.get('LING_GPU_PREFILL','0') == '1':
    assert pool <= 64 and cap <= 96 and int(os.environ.get('LING_STOP_GIB','20')) >= 26, 'GPU prefill diagnostic requires pool <=64, cap <=96 and stop >=26 GiB'
s=Path(sys.argv[1]).read_text()
s=re.sub(r'^pin_file = .*$', 'pin_file = '+json.dumps(str(Path(sys.argv[3]).resolve())), s, flags=re.M)
Path(sys.argv[2]).write_text(s)
CONFIG
minimum_kb=120500000
[[ -x "$worktree/build-ling/moestream" ]]
[[ -s "$profile" ]]
[[ -r /home/ray/vllm-workspace/ling-3.1-flash/Ling-3.1-flash-AD-IQ1_S-00001-of-00004.gguf ]]
grep -q 'MemorySwapMax=0' "$worktree/scripts/run_capped.sh"
exec > >(tee "$log") 2>&1
exec 9>/tmp/moestream-host-admission.lock
if ! flock -n 9; then
  echo '[PREFLIGHT] shared admission lock is held; not starting.'
  exit 31
fi
python3 - "$minimum_kb" <<'PY'
from pathlib import Path
import os, sys, time
minimum_kb=int(sys.argv[1])
# Ignore this script, the checker, and their launcher ancestors. MoEStream
# acquires the same admission lock before model initialization.
ancestors=set()
pid=os.getpid()
while pid > 1:
    ancestors.add(pid)
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split()
        pid=int(fields[3])
    except (OSError,ValueError,IndexError): break
names={'moestream','llama-server','llama-cli','llama-bench','llama-mtmd-cli',
       'llama-completion','llama-simple','llama-parallel','llama-quantize',
       'llama-gguf-split','llama-gguf-surgery','abrun.sh',
       'run_split_queue58.sh','run_dsa_graph2g.sh'}
builders={'cmake','make','gmake','ninja','nvcc','cc1plus','cc1','c++','g++','clang++'}
blocked=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name) in ancestors: continue
    try:
        comm=(p/'comm').read_text().strip().lower()
        args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    except (OSError,ProcessLookupError): continue
    joined=' '.join(args).lower()
    if comm in names or (comm.startswith('llama-')):
        blocked.append(f'{p.name} {comm}: {joined[:260]}')
    elif comm in builders and any(x in joined for x in ('/home/ray/projects/moestream','/home/ray/projects/llama.cpp')):
        blocked.append(f'{p.name} {comm}: {joined[:260]}')
    elif comm in {'bash','sh','systemd-run','flock'} and any(x in joined for x in ('build-ling/moestream','build_g7/moestream','llama-server','llama-cli','llama-bench','abrun.sh','run_split_queue58.sh','run_dsa_graph2g.sh')):
        blocked.append(f'{p.name} {comm}: {joined[:260]}')
if blocked:
    print('[PREFLIGHT] model/build/sweep process detected; waiting for it to finish:')
    print('\n'.join(blocked))
    raise SystemExit(32)
print('[PREFLIGHT] no MoEStream/llama.cpp model, build, or sweep process detected')
for i in range(1,4):
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
    if available < minimum_kb:
        print(f'[PREFLIGHT] MemAvailable={available/1e6:.2f} GB < {minimum_kb/1e6:.2f} GB; not starting')
        raise SystemExit(33)
    print(f'[PREFLIGHT] sample {i}/3 MemAvailable={available/1e6:.2f} GB')
    if i<3: time.sleep(1)
print('[PREFLIGHT] host swap may be in use by unrelated processes; this run is confined to MemorySwapMax=0')
PY
flock -u 9
export MOESTREAM_LING_FULL_GRAPH=1
export MOESTREAM_FULL_GRAPH_RESIDENT_GB=3
export MOESTREAM_FULL_GRAPH_USAGE_FILE="$profile"
export MOESTREAM_GLM5NEXT_DYNAMIC_CACHE=0
export MOESTREAM_GLM5NEXT_GPU1_HOT_GIB=12.5
export MOESTREAM_GLM5NEXT_GPU1_HOT_USAGE="$profile"
export MOESTREAM_GLM5NEXT_GPU_ASSIST=5
export MOESTREAM_GLM5NEXT_ASSIST2_SLOTS=2
export MOESTREAM_GLM5NEXT_CPU_THREADS=10
export MOESTREAM_GLM5NEXT_EARLY_READY=1
export MOESTREAM_LING_OFFLOAD_FIRST_LAYER=4
export MOESTREAM_LING_OFFLOAD_MODEL=/home/ray/vllm-workspace/ling-3.1-flash/Ling-3.1-flash-AD-IQ1_S-00001-of-00004.gguf
export MOESTREAM_LING_OFFLOAD_USAGE="$profile"
export MOESTREAM_LING_OFFLOAD_DDR_GIB=${LING_DDR_GIB:-72}
export MOESTREAM_LING_OFFLOAD_PINNED=0.2
export MOESTREAM_LING_OFFLOAD_BYTE_CAP=1
export MOESTREAM_LING_OFFLOAD_PIN=1
export MOESTREAM_LING_OFFLOAD_ASYNC=1
unset MOESTREAM_LING_NATIVE_GRAPH MOESTREAM_LING_HYBRID_CUDA_GRAPH MOESTREAM_LING_CPU_COLD_MISSES MOESTREAM_GLM5NEXT_OFFLOAD_FIRST_LAYER MOESTREAM_GLM5NEXT_OFFLOAD_MODEL MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB MOESTREAM_GLM5NEXT_OFFLOAD_POLICY MOESTREAM_GLM5NEXT_OFFLOAD_PINNED MOESTREAM_GLM5NEXT_OFFLOAD_PROBATION MOESTREAM_GLM5NEXT_OFFLOAD_BYTE_CAP MOESTREAM_GLM5NEXT_OFFLOAD_PIN MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC MOESTREAM_GLM5NEXT_OFFLOAD_USAGE MOESTREAM_GLM5NEXT_OFFLOAD_THP MOESTREAM_GLM5NEXT_OFFLOAD_STORE MOESTREAM_FULL_GRAPH_USE_RAM_CACHE MOESTREAM_LING_FULL_GRAPH_DDR_PREFETCH_GIB
printf '[RUN] prompt=Hi there; ngen=32; temp=0; config=%s; usage=%s; DDR=%s GiB; GPU hot=3+12.5 GiB; assist=5+2; CPU threads=10\n' "$config" "$profile" "${LING_DDR_GIB:-72}"
export MOESTREAM_GLM5NEXT_ROUTE_TRACE="${log%.log}.routes"
export MOESTREAM_GENERATED_TOKEN_FILE="${log%.log}.tokens"
unset MOESTREAM_TEACHER_FORCE_TOKEN_FILE MOESTREAM_LING_PREFILL_OP_OFFLOAD GGML_OP_OFFLOAD_MIN_BATCH
if [[ -n "${LING_TOKEN_FILE:-}" ]]; then export MOESTREAM_TEACHER_FORCE_TOKEN_FILE="$LING_TOKEN_FILE"; fi
export MOESTREAM_GLM5NEXT_PREFILL_OP_OFFLOAD=${LING_GPU_PREFILL:-0}
if [[ "${LING_GPU_PREFILL:-0}" == 1 ]]; then
  export GGML_OP_OFFLOAD_MIN_BATCH=2 MOESTREAM_GLM5NEXT_ALLOC_FLOOR_MIB=768
fi
exec "$worktree/scripts/run_capped.sh" --limit-gb "${LING_CAP_GIB:-110}" "$worktree/build-ling/moestream" --config "$config" --ngen 32 --temp 0 < "$worktree/bench/ling/reference-audit/prompt.txt"
