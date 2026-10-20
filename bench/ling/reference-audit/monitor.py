import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name('run.sh'))
BINARY = str(ROOT / 'build-ling/moestream')
PREFIX = sys.argv[2] if len(sys.argv) > 2 else '/tmp/ling-reference-monitor-' + str(time.time_ns())
OUT = Path(PREFIX + '.memory.jsonl')
FLOOR_KIB = int(os.environ.get('LING_STOP_GIB', '20')) * 1024 * 1024
CAP = int(os.environ.get('LING_CAP_GIB', '110')) * 1024**3

with open(PREFIX + '.driver.log', 'w') as driver:
    proc = subprocess.Popen([SCRIPT], stdout=driver, stderr=subprocess.STDOUT)
    model_pid = None
    cgroup = None
    last_gpu = 0
    stopped = False
    with OUT.open('w', buffering=1) as out:
        while proc.poll() is None:
            available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
            if model_pid is None:
                for p in Path('/proc').iterdir():
                    if not p.name.isdigit():
                        continue
                    try:
                        if os.readlink(p / 'exe') != BINARY:
                            continue
                        parent = int(p.name)
                        seen = set()
                        while parent != proc.pid and parent > 1 and parent not in seen:
                            seen.add(parent)
                            parent = int(Path(f'/proc/{parent}/stat').read_text().rsplit(')', 1)[1].split()[1])
                        if parent != proc.pid:
                            continue
                        model_pid = int(p.name)
                        rel = next(x.split(':', 2)[2] for x in (p / 'cgroup').read_text().splitlines() if x.startswith('0:'))
                        cgroup = Path('/sys/fs/cgroup') / rel.lstrip('/')
                        print('Monitoring model PID', model_pid, 'cgroup', cgroup, flush=True)
                        break
                    except (OSError, StopIteration):
                        continue
            sample = {'time': time.time(), 'available_kib': available}
            if cgroup is not None:
                for name in ['memory.current', 'memory.peak', 'memory.max', 'memory.swap.current', 'memory.swap.max', 'memory.events']:
                    try:
                        value = (cgroup / name).read_text().strip()
                        sample[name] = int(value) if value.isdigit() else value
                    except OSError:
                        pass
                unsafe = available < FLOOR_KIB
                unsafe |= sample.get('memory.swap.current', 0) != 0
                unsafe |= sample.get('memory.swap.max', 0) != 0
                limit = sample.get('memory.max')
                unsafe |= not isinstance(limit, int) or limit > CAP
                if unsafe and not stopped:
                    print('Stopping own model at memory guard:', json.dumps(sample), flush=True)
                    try:
                        if os.readlink(f'/proc/{model_pid}/exe') == BINARY:
                            os.kill(model_pid, signal.SIGKILL)
                    except OSError:
                        pass
                    stopped = True
                if time.monotonic() - last_gpu > 8:
                    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True)
                    sample['gpu'] = gpu.stdout.strip()
                    last_gpu = time.monotonic()
            out.write(json.dumps(sample) + '\n')
            time.sleep(0.5)
    status = proc.wait()
    print('Benchmark exit', status, 'memory samples', OUT, flush=True)
    raise SystemExit(status)
