#!/usr/bin/env bash
set -euo pipefail
worktree=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
profile=${LING_USAGE_FILE:-$worktree/bench/ling/reference-audit/usage-hi-there-robust4.pin}
# Empty by default: use an explicit prompt-matched profile for prefill pinning.
prefill_profile=${LING_PREFILL_USAGE_FILE-}
prompt_file=${LING_PROMPT_FILE:-$worktree/bench/ling/reference-audit/prompt.txt}
context_size=${LING_CONTEXT_SIZE:-64}
freeze_usage_stats=${LING_FREEZE_USAGE_STATS:-1}
log=${LING_LOG:-/tmp/ling-reference-$(date -u +%Y%m%dT%H%M%S).log}
config=$(mktemp /tmp/ling-reference-config.XXXXXX.toml)
python3 - "$worktree/bench/ling/reference-audit/config.toml" "$config" "$profile" "$prefill_profile" "$freeze_usage_stats" "$context_size" <<'CONFIG'
from pathlib import Path
import json, os, re, sys
pool=int(os.environ.get('LING_DDR_GIB','72')); cap=int(os.environ.get('LING_CAP_GIB','95'))
stop=float(os.environ.get('LING_STOP_GIB','20'))
cpu_threads=int(os.environ.get('LING_CPU_THREADS','10'))
offload_pinned=float(os.environ.get('LING_OFFLOAD_PINNED','0.5'))
gpu_assist=int(os.environ.get('LING_GPU_ASSIST','3'))
assist2_slots=int(os.environ.get('LING_ASSIST2_SLOTS','1'))
early_ready=int(os.environ.get('LING_EARLY_READY','1'))
gpu0_cache=float(os.environ.get('LING_GPU0_CACHE_GIB','3.0'))
gpu1_cache=float(os.environ.get('LING_GPU1_CACHE_GIB','12.5'))
pinned_vram_capacity=int(os.environ.get('LING_PINNED_VRAM_CAPACITY','3700'))
assert pool in (64, 68, 72) and 0 < cap <= 95 and stop >= 20, 'No-copy graph runs require DDR pool <=72 GiB, cgroup cap <=95 GiB, and at least a 20 GiB available-memory stop'
assert 0.0 <= offload_pinned <= 1.0, 'LING_OFFLOAD_PINNED must be between 0 and 1'
assert 0 <= gpu_assist <= 8 and 0 <= assist2_slots <= 8, 'GPU assist slot counts must be between 0 and 8'
assert early_ready in (0, 1), 'LING_EARLY_READY must be 0 or 1'
assert 0.0 < gpu0_cache <= 8.0 and 0.0 < gpu1_cache <= 14.0, 'GPU cache requests must be in range (0,8] + (0,14] GiB'
assert 0 <= pinned_vram_capacity <= 6000, 'LING_PINNED_VRAM_CAPACITY must be between 0 and 6000'
s=Path(sys.argv[1]).read_text()
context_size=int(sys.argv[6])
assert 64 <= context_size <= 131072, 'LING_CONTEXT_SIZE must be between 64 and 131072'
s=re.sub(r'^pin_file = .*$', 'pin_file = '+json.dumps(str(Path(sys.argv[3]).resolve())), s, flags=re.M)
s=re.sub(r'^prefill_pin_file = .*$', 'prefill_pin_file = '+json.dumps(str(Path(sys.argv[4]).resolve()) if sys.argv[4] else ''), s, flags=re.M)
s=re.sub(r'^freeze_usage_stats = .*$', 'freeze_usage_stats = '+('true' if sys.argv[5].lower() not in ('0','false','no') else 'false'), s, flags=re.M)
s,n_ctx=re.subn(r'^context_size = \d+$', 'context_size = '+str(context_size), s, flags=re.M)
assert n_ctx == 1, f'Expected one context_size entry, got {n_ctx}'
s=re.sub(r'^omp_max_threads = .*$', 'omp_max_threads = '+str(cpu_threads), s, flags=re.M)
cache_values=iter((gpu0_cache, gpu1_cache))
s,n_cache=re.subn(r'^moe_cache_gb = .*$', lambda _: 'moe_cache_gb = '+str(next(cache_values)), s, flags=re.M)
assert n_cache == 2, f'Expected two GPU moe_cache_gb entries, got {n_cache}'
s,n_pin=re.subn(r'^pinned_vram_capacity = .*$', 'pinned_vram_capacity = '+str(pinned_vram_capacity), s, flags=re.M)
assert n_pin == 1, f'Expected one pinned_vram_capacity entry, got {n_pin}'
Path(sys.argv[2]).write_text(s)
CONFIG
minimum_kb=120500000
[[ -x "$worktree/build-ling/moestream" ]]
[[ -r "$prompt_file" && -s "$prompt_file" ]]
if [[ ! "$freeze_usage_stats" =~ ^(0|false|no)$ ]]; then
  [[ -s "$profile" ]]
fi
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
export MOESTREAM_GLM5NEXT_FULL_GRAPH_PREFILL=1
export MOESTREAM_FULL_GRAPH_USAGE_FILE="$profile"
export MOESTREAM_GLM5NEXT_DYNAMIC_CACHE=0
export MOESTREAM_GLM5NEXT_GPU_ASSIST=${LING_GPU_ASSIST:-3}
export MOESTREAM_GLM5NEXT_ASSIST2_SLOTS=${LING_ASSIST2_SLOTS:-1}
export MOESTREAM_GLM5NEXT_CPU_THREADS=${LING_CPU_THREADS:-10}
export MOESTREAM_GLM5NEXT_EARLY_READY=${LING_EARLY_READY:-1}
export MOESTREAM_LING_OFFLOAD_FIRST_LAYER=4
export MOESTREAM_LING_OFFLOAD_MODEL=/home/ray/vllm-workspace/ling-3.1-flash/Ling-3.1-flash-AD-IQ1_S-00001-of-00004.gguf
export MOESTREAM_LING_OFFLOAD_USAGE="$profile"
export MOESTREAM_LING_OFFLOAD_DDR_GIB=${LING_DDR_GIB:-72}
export MOESTREAM_LING_OFFLOAD_PINNED=${LING_OFFLOAD_PINNED:-0.5}
export MOESTREAM_LING_OFFLOAD_BYTE_CAP=1
export MOESTREAM_LING_OFFLOAD_PIN=1
export MOESTREAM_LING_OFFLOAD_ASYNC=1
# Ling's batched prefill needs the patched negative-id skip once the prompt
# grows past the two-token smoke case: the valid-dummy fallback aliases many
# tokens onto expert 0 and ggml's batched id compaction then over-runs that
# expert's row range.  Keep the historical short-prompt behavior by default;
# long-prompt runs (context > 64) opt into the patched path.  Override with
# LING_BATCHED_NEGATIVE_SKIP=0/1 for an A/B.
if [[ -n "${LING_BATCHED_NEGATIVE_SKIP:-}" ]]; then
  export MOESTREAM_GLM5NEXT_BATCHED_NEGATIVE_SKIP="$LING_BATCHED_NEGATIVE_SKIP"
elif (( context_size > 64 )); then
  export MOESTREAM_GLM5NEXT_BATCHED_NEGATIVE_SKIP=1
else
  export MOESTREAM_GLM5NEXT_BATCHED_NEGATIVE_SKIP=0
fi
unset MOESTREAM_LING_NATIVE_GRAPH MOESTREAM_LING_HYBRID_CUDA_GRAPH MOESTREAM_LING_CPU_COLD_MISSES MOESTREAM_GLM5NEXT_OFFLOAD_FIRST_LAYER MOESTREAM_GLM5NEXT_OFFLOAD_MODEL MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB MOESTREAM_GLM5NEXT_OFFLOAD_POLICY MOESTREAM_GLM5NEXT_OFFLOAD_PINNED MOESTREAM_GLM5NEXT_OFFLOAD_PROBATION MOESTREAM_GLM5NEXT_OFFLOAD_BYTE_CAP MOESTREAM_GLM5NEXT_OFFLOAD_PIN MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC MOESTREAM_GLM5NEXT_OFFLOAD_USAGE MOESTREAM_GLM5NEXT_OFFLOAD_THP MOESTREAM_GLM5NEXT_OFFLOAD_STORE MOESTREAM_GLM5NEXT_GPU1_HOT_GIB MOESTREAM_GLM5NEXT_GPU1_HOT_USAGE MOESTREAM_FULL_GRAPH_RESIDENT_GB MOESTREAM_FULL_GRAPH_USE_RAM_CACHE MOESTREAM_LING_FULL_GRAPH_DDR_PREFETCH_GIB
prompt_sha256=$(sha256sum -- "$prompt_file" | awk '{print $1}')
printf '[RUN] prompt_file=%s; prompt_sha256=%s; context=%s; ngen=32; temp=0; config=%s; usage=%s; DDR=%s GiB OffloadStore; GPU cache=%s+%s GiB; pinned VRAM slots=%s; pinned DDR fraction=%s; GPU assist slots=%s+%s; early-ready=%s; batched-negative-skip=%s; MoE tensors alias cache; CPU threads=%s\n' "$prompt_file" "$prompt_sha256" "$context_size" "$config" "$profile" "${LING_DDR_GIB:-72}" "${LING_GPU0_CACHE_GIB:-3.0}" "${LING_GPU1_CACHE_GIB:-12.5}" "${LING_PINNED_VRAM_CAPACITY:-3700}" "${LING_OFFLOAD_PINNED:-0.5}" "${LING_GPU_ASSIST:-3}" "${LING_ASSIST2_SLOTS:-1}" "${LING_EARLY_READY:-1}" "${MOESTREAM_GLM5NEXT_BATCHED_NEGATIVE_SKIP}" "${LING_CPU_THREADS:-10}"
export MOESTREAM_GLM5NEXT_ROUTE_TRACE="${log%.log}.routes"
export MOESTREAM_GENERATED_TOKEN_FILE="${log%.log}.tokens"
unset MOESTREAM_TEACHER_FORCE_TOKEN_FILE MOESTREAM_LING_PREFILL_OP_OFFLOAD GGML_OP_OFFLOAD_MIN_BATCH
if [[ -n "${LING_TOKEN_FILE:-}" ]]; then export MOESTREAM_TEACHER_FORCE_TOKEN_FILE="$LING_TOKEN_FILE"; fi
if [[ "$freeze_usage_stats" =~ ^(0|false|no)$ ]]; then
  export MOESTREAM_GLM5NEXT_PREFILL_ROUTE_CAPTURE=1
else
  unset MOESTREAM_GLM5NEXT_PREFILL_ROUTE_CAPTURE
fi
export MOESTREAM_GLM5NEXT_PREFILL_OP_OFFLOAD=0
exec "$worktree/scripts/run_capped.sh" --limit-gb "${LING_CAP_GIB:-95}" "$worktree/build-ling/moestream" --config "$config" --ngen 32 --temp 0 < "$prompt_file"
