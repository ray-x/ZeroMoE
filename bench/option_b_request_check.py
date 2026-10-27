#!/usr/bin/env python3
"""Check free-running code/creative/code requests in one engine lifetime.

Natural stop handling is enabled. Reference verification and output capture
are correctness evidence, with no throughput claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from option_b_global_pool_check import ROOT, A, setting
from option_b_topic_check import PROMPTS, wait_idle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--ngen', type=int, default=512)
    p.add_argument('--wait-idle', type=int, default=300)
    args = p.parse_args()
    engine, out = args.engine.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True)
    requests = [PROMPTS['code'], PROMPTS['creative'], PROMPTS['code']]
    (out / 'requests.txt').write_text('\n'.join(requests) + '\n:q\n')
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (engine, A)}
    env = {k: v for k, v in os.environ.items() if not k.startswith('ZEROMOE_')}
    for name in ('OMP_WAIT_POLICY', 'GOMP_SPINCOUNT', 'OMP_PROC_BIND', 'OMP_PLACES'):
        env.pop(name, None)
    env.update(CUDA_VISIBLE_DEVICES='0', GGML_CUDA_BF16_MMVF_CUBLAS='0',
        ZEROMOE_QWEN_CPU_THREADS='10', ZEROMOE_BENCH_IGNORE_STOP='0',
        ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT='1', ZEROMOE_QWEN_EARLY_READY_RESIDENT_TABLES='1',
        ZEROMOE_QWEN_EARLY_READY_PREDICT='1')
    results = {}
    (out / 'manifest.json').write_text(json.dumps({'hashes': hashes, 'ngen': args.ngen,
        'requests': requests, 'timing_valid': False}, indent=2) + '\n')
    for arm in ('baseline', 'staging'):
        wait_idle(args.wait_idle)
        for path, expected in hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'input changed: {path}')
        log, config = out / f'{arm}.engine.log', out / f'{arm}.toml'
        config.write_text(setting(A.read_text(), 'log_path', json.dumps(str(log))))
        runtime = dict(env)
        if arm == 'staging':
            runtime.update(ZEROMOE_QWEN_EARLY_READY_STAGE_LAYER='all',
                           ZEROMOE_QWEN_EARLY_READY_STAGE_VERIFY='1')
        print(f'Running {arm}: three natural-stop requests', flush=True)
        with (out / f'{arm}.stdout').open('w') as output:
            subprocess.run([str(engine), '--graph-device-expert-layers', '0', '--graph-device-expert-trailing', '1', '--config', str(config), '--predict', str(args.ngen)],
                input=(out / 'requests.txt').read_text(), text=True, cwd=ROOT, env=runtime,
                stdout=output, stderr=subprocess.STDOUT, check=True)
        contents = log.read_text()
        if any(x in contents for x in ('layered decode remains active', 'using layered decode',
                                       'refusing graph allocator reserve', 'CUDA error')):
            raise RuntimeError(f'{arm}: graph fallback or CUDA failure')
        stdout = re.sub(r'\x1b\[[0-9;]*m', '', (out / f'{arm}.stdout').read_text())
        responses = [part.split('[Generation Complete]', 1)[0].strip()
                     for part in stdout.split('Output:')[1:]]
        if len(responses) != len(requests) or stdout.count('[Generation Complete]') != len(requests):
            raise RuntimeError(f'{arm}: incomplete request sequence')
        for index, response in enumerate(responses, 1):
            (out / f'{arm}.response_{index}.txt').write_text(response + '\n')
        results[arm] = {'completed_requests': len(responses),
                       'first_vs_third_equal': responses[0] == responses[2],
                       'reference_diagnostics': contents.count('[OPTION-B-STAGE-PREDICT]')}
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({arm: results[arm]}), flush=True)


if __name__ == '__main__':
    main()
