#!/usr/bin/env python3
"""Compare the opt-in Layer-13 bridge on an identical token history.

The checkpoint must pass gguf_tool verify before running. Both arms use the
same binary/config; only MOESTREAM_QWEN_EARLY_READY_LAYER13 changes.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import resource
import statistics
import subprocess
import time

from option_b_global_pool_check import ROOT, A, setting
from compare_decode_logits import compare


def competing_cpu(exclude=None):
    names = {'test_components', 'cc1plus', 'cc1', 'nvcc', 'ptxas', 'cudafe++',
             'collect2', 'ollama', 'vllm'}
    rows = subprocess.check_output(['ps', '-eo', 'pid=,comm=,args='], text=True).splitlines()
    busy = []
    for row in rows:
        fields = row.strip().split(None, 2)
        if len(fields) < 2:
            continue
        pid, name = int(fields[0]), fields[1]
        if pid != exclude and (name in names or name.startswith(
                ('moestream', 'llama-', 'exl3_', 'qwen_cpu_')) or '_bench' in name):
            busy.append(row.strip())
    return busy


def competing_gpu(exclude=None):
    # Other adapters share CPU and PCIe resources, too.
    probe = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
        text=True, stdout=subprocess.PIPE, check=False)
    # nvidia-smi returns 9 when the query has no active compute processes.
    # That is the idle state the admission check is trying to detect.
    if probe.returncode == 9:
        return []
    if probe.returncode:
        raise subprocess.CalledProcessError(probe.returncode, probe.args)
    rows = probe.stdout.splitlines()
    return [row for row in rows if row.strip() and int(row.split(',', 1)[0]) != exclude]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--prompt-ids', type=Path, required=True)
    p.add_argument('--continuation', type=Path, required=True)
    p.add_argument('--reps', type=int, default=6)
    p.add_argument('--threads', type=int, default=10)
    p.add_argument('--wait-idle', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--all-split', action='store_true')
    p.add_argument('--allow-busy-correctness', action='store_true')
    p.add_argument('--dump-logits', action='store_true')
    p.add_argument('--free-running', action='store_true')
    p.add_argument('--baseline-env', action='append', default=[], metavar='NAME=VALUE')
    p.add_argument('--candidate-env', action='append', default=[], metavar='NAME=VALUE')
    p.add_argument('--third-env', action='append', default=[], metavar='NAME=VALUE',
                   help='optional third arm; six permutations balance its order')
    args = p.parse_args()
    for value in args.baseline_env + args.candidate_env + args.third_env:
        if '=' not in value:
            p.error('environment overrides require NAME=VALUE')
    if args.third_env and args.reps % 6 and not args.dump_logits:
        p.error('three-arm comparisons require a multiple of six repeats')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args.allow_busy_correctness and not args.dump_logits:
        p.error('--allow-busy-correctness requires --dump-logits')
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    engine = args.engine.resolve()
    ngen = len(args.continuation.read_text().split())
    if ngen < 4 or args.reps < 1:
        p.error('need at least four continuation tokens and one repeat')
    env = {k: v for k, v in os.environ.items() if not k.startswith('MOESTREAM_')}
    for key in ('OMP_WAIT_POLICY', 'GOMP_SPINCOUNT', 'OMP_PROC_BIND', 'OMP_PLACES'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='0', GGML_CUDA_BF16_MMVF_CUBLAS='0',
               MOESTREAM_QWEN4EXP_FULL_GRAPH='1',
               MOESTREAM_QWEN4EXP_FULL_GRAPH_DECODE='1',
               MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_LAYERS='0',
               MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_TRAILING='1',
               MOESTREAM_QWEN_CPU_THREADS=str(args.threads),
               MOESTREAM_BENCH_IGNORE_STOP='1', MOESTREAM_TOKEN_LOOP_DIAG='1',
               MOESTREAM_PROMPT_TOKEN_INPUT_FILE=str(args.prompt_ids.resolve()))
    env.pop('MOESTREAM_DECODE_LOGITS_DUMP', None)
    env.pop('MOESTREAM_TEACHER_FORCE_TOKEN_FILE', None)
    env.pop('MOESTREAM_QWEN4EXP_GRAPH_LAYER_TIMING', None)
    if not args.free_running:
        env['MOESTREAM_TEACHER_FORCE_TOKEN_FILE'] = str(args.continuation.resolve())
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in (engine, A, args.prompt_ids, args.continuation)}
    def check_inputs():
        for name, expected in hashes.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'benchmark input changed during run: {name}')
    manifest = {'hashes': hashes, 'threads': args.threads, 'ngen': ngen,
                'reps': args.reps, 'free_running': args.free_running,
                'dump_logits': args.dump_logits, 'all_split': args.all_split,
                'allow_busy_correctness': args.allow_busy_correctness,
                'admission_version': 2,
                'baseline_env': args.baseline_env, 'candidate_env': args.candidate_env,
                'third_env': args.third_env,
                'admission': []}
    rows = []
    if args.resume:
        old = json.loads((out / 'manifest.json').read_text())
        for key in ('hashes', 'threads', 'ngen', 'reps', 'free_running', 'dump_logits', 'all_split'):
            if old.get(key, False) != manifest[key]:
                raise RuntimeError(f'resume settings differ: {key}')
        for key in ('baseline_env', 'candidate_env', 'third_env'):
            if old.get(key, []) != manifest[key]:
                raise RuntimeError(f'resume settings differ: {key}')
        if old.get('admission_version', 1) != manifest['admission_version']:
            raise RuntimeError('resume admission monitoring differs; preserve the old batch and use a new directory')
        if old.get('allow_busy_correctness', False) != args.allow_busy_correctness:
            if not (args.dump_logits and args.allow_busy_correctness):
                raise RuntimeError('resume admission policy differs')
            old['allow_busy_correctness'] = True
            old.setdefault('admission_updates', []).append('allow CPU contention for correctness dump only')
        manifest = old
        if 'interference' in manifest:
            attempt = {'admission': manifest['admission'][-1],
                       'interference': manifest.pop('interference')}
            manifest.setdefault('discarded_attempts', []).append(attempt)
        rows = json.loads((out / 'results.json').read_text()) if (out / 'results.json').exists() else []
    arms = ('baseline', 'early', 'extra') if args.third_env else ('baseline', 'early')
    orders = list(itertools.permutations(arms))
    for rep in range(1, args.reps + 1):
        for arm in orders[(rep - 1) % len(orders)]:
            if any(row['arm'] == arm and row['rep'] == rep for row in rows):
                print(f'Reusing valid {arm}_{rep}', flush=True)
                continue
            deadline = time.monotonic() + args.wait_idle
            while True:
                busy = competing_cpu()
                if not busy or args.allow_busy_correctness:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'host busy before {arm}_{rep}: {busy}')
                time.sleep(5)
            processes = '\n'.join(competing_gpu())
            cpu = '\n'.join(competing_cpu())
            if (processes or cpu) and not args.allow_busy_correctness:
                raise RuntimeError(f'competing inference: {processes or cpu}')
            manifest['admission'].append({'arm': arm, 'rep': rep,
                                          'gpu_processes': processes, 'cpu_inference': cpu})
            check_inputs()
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            tag = f'{arm}_{rep}'
            log = out / f'{tag}.engine.log'
            cfg = out / f'{tag}.toml'
            if args.resume:
                discard = out / 'discarded'
                discard.mkdir(exist_ok=True)
                for suffix in ('engine.log', 'stdout', 'tokens', 'toml', 'logits'):
                    previous = out / f'{tag}.{suffix}'
                    if previous.exists():
                        previous.rename(discard / f'{previous.name}.{time.time_ns()}')
            cfg.write_text(setting(A.read_text(), 'log_path', json.dumps(str(log))))
            runtime = dict(env,
                MOESTREAM_QWEN_EARLY_READY_LAYER13='1' if arm != 'baseline' and not args.all_split else '0',
                MOESTREAM_QWEN_EARLY_READY_ALL_SPLIT='1' if arm != 'baseline' and args.all_split else '0')
            overrides = {'baseline': args.baseline_env, 'early': args.candidate_env,
                         'extra': args.third_env}[arm]
            for value in overrides:
                name, value = value.split('=', 1)
                runtime[name] = value
            if args.dump_logits:
                runtime['MOESTREAM_DECODE_LOGITS_DUMP'] = str(out / f'{tag}.logits')
            runtime['MOESTREAM_GENERATED_TOKEN_FILE'] = str(out / f'{tag}.tokens')
            print(f'Running {tag}', flush=True)
            with (out / f'{tag}.stdout').open('w') as output:
                process = subprocess.Popen([str(engine), '--config', str(cfg), '--ngen', str(ngen)],
                               stdin=subprocess.PIPE, text=True, cwd=ROOT, env=runtime,
                               stdout=output, stderr=subprocess.STDOUT)
                prompt = 'Explain briefly the key concepts of neural networks and gradient descent in plain English.\n:q\n'
                interference = []
                while True:
                    try:
                        process.communicate(input=prompt, timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        prompt = None
                        interference.extend(competing_cpu(process.pid))
                        interference.extend(competing_gpu(process.pid))
                if interference:
                    manifest['interference'] = sorted(set(interference))
                    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
                    if not args.allow_busy_correctness:
                        raise RuntimeError(f'{tag}: host contention during run: {manifest["interference"]}')
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, process.args)
            check_inputs()
            contents = log.read_text()
            failures = ('using layered decode', 'layered decode remains active',
                        'refusing graph allocator reserve', 'CUDA error')
            if any(message in contents for message in failures):
                raise RuntimeError(f'{tag}: graph fallback or CUDA failure')
            times = [(int(step), float(total), float(core)) for step, total, core in re.findall(
                r'TOKEN-LOOP-DIAG.*step=(\d+) iter_total=([\d.]+)ms.*core\(hc\+forward\)=([\d.]+)ms', contents)]
            steady = [(total, core) for step, total, core in times if step >= 3]
            if [step for step, _, _ in times] not in (list(range(ngen)), list(range(ngen - 1))):
                raise RuntimeError(f'{tag}: incomplete timing')
            if arm != 'baseline':
                exercised = ('[OPTION-B-EARLY-READY] calls=' in contents or
                             '[QWEN4EXP-PARTIAL-GRAPH]' in contents)
                if not exercised:
                    raise RuntimeError(f'{tag}: graph bridge not exercised')
            totals, cores = zip(*steady)
            row = {'arm': arm, 'rep': rep, 'tps': 1000 / statistics.mean(totals),
                   'loop_ms': statistics.mean(totals), 'core_ms': statistics.mean(cores)}
            rows.append(row)
            (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    if args.dump_logits and not args.free_running:
        baseline, early = out / 'baseline_1.logits', out / 'early_1.logits'
        result = compare(baseline, early, expected_steps=ngen)
        (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
        print('Logits identical:', baseline.read_bytes() == early.read_bytes(), flush=True)
        if args.third_env:
            result = compare(early, out / 'extra_1.logits', expected_steps=ngen)
            (out / 'third_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    summary = {arm: statistics.mean(r['tps'] for r in rows if r['arm'] == arm)
               for arm in arms}
    summary['gain_pct'] = 100 * (summary['early'] / summary['baseline'] - 1)
    pairs = [{r['arm']: r['tps'] for r in rows if r['rep'] == rep}
             for rep in range(1, args.reps + 1)]
    summary['paired_gain_pct'] = [100 * (x['early'] / x['baseline'] - 1) for x in pairs]
    summary['paired_wins'] = sum(x['early'] > x['baseline'] for x in pairs)
    if args.third_env:
        summary['extra_gain_pct'] = 100 * (summary['extra'] / summary['early'] - 1)
        summary['extra_vs_baseline_gain_pct'] = 100 * (summary['extra'] / summary['baseline'] - 1)
        summary['extra_paired_gain_pct'] = [100 * (x['extra'] / x['early'] - 1) for x in pairs]
        summary['extra_paired_wins'] = sum(x['extra'] > x['early'] for x in pairs)
    summary['timing_valid'] = not args.dump_logits and not args.allow_busy_correctness
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
