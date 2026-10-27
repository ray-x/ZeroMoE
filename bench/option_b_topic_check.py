#!/usr/bin/env python3
"""Capture topical histories, then verify next-layer prediction and staging.

Checkpoint verification is a prerequisite. Dump/trace/reference-verification
runs are correctness evidence only; use the separate six-pair timing harness
for throughput acceptance. Every generated artifact is retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from option_b_global_pool_check import ROOT, A, setting
from analyze_qwen_route_prediction import analyze


PROMPTS = {
    'code': ('Write a Python function that merges overlapping intervals. Explain '
             'how it handles empty input, touching intervals and unsorted input, '
             'then give two small examples.'),
    'creative': ('Write a short scene in which a lighthouse keeper discovers that '
                 'the light is sending messages from tomorrow. Use concrete '
                 'details and dialogue, and end with a difficult choice.'),
}


def wait_idle(seconds):
    deadline = time.monotonic() + seconds
    while True:
        busy = []
        for name in ('zeromoe', 'zeromoe_tests', 'test_components', 'llama-server',
                     'llama-cli', 'llama-bench', 'cc1plus', 'nvcc'):
            busy.extend(subprocess.run(['pgrep', '-x', '-a', name], text=True,
                                       capture_output=True).stdout.splitlines())
        gpu = subprocess.check_output(['nvidia-smi', '--id=0',
            '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
        if not busy and not gpu:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f'host busy: {busy}, GPU: {gpu}')
        time.sleep(5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--topic', choices=PROMPTS, action='append')
    p.add_argument('--ngen', type=int, default=64)
    p.add_argument('--wait-idle', type=int, default=300)
    p.add_argument('--stage-layer', default='all')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--free-running', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    engine = args.engine.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ZEROMOE_')}
    for name in ('OMP_WAIT_POLICY', 'GOMP_SPINCOUNT', 'OMP_PROC_BIND', 'OMP_PLACES'):
        env.pop(name, None)
    env.update(CUDA_VISIBLE_DEVICES='0', GGML_CUDA_BF16_MMVF_CUBLAS='0',
                ZEROMOE_QWEN_CPU_THREADS='10',
        ZEROMOE_BENCH_IGNORE_STOP='1', ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT='1',
        ZEROMOE_QWEN_EARLY_READY_RESIDENT_TABLES='1', ZEROMOE_QWEN_EARLY_READY_PREDICT='1')
    manifest = {
        'engine': str(engine), 'engine_sha256': hashlib.sha256(engine.read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256(A.read_bytes()).hexdigest(),
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'timing_valid': False, 'prompts': PROMPTS}
    if args.resume:
        previous = json.loads((out / 'manifest.json').read_text())
        for key in ('engine_sha256', 'config_sha256', 'prompts'):
            if previous[key] != manifest[key]:
                raise RuntimeError(f'resume differs: {key}')
        for key in ('ngen', 'stage_layer', 'verify', 'free_running', 'topic'):
            if previous['args'].get(key) != manifest['args'].get(key):
                raise RuntimeError(f'resume differs: {key}')
    else:
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for topic in args.topic or PROMPTS:
        directory = out / topic
        directory.mkdir(exist_ok=args.resume)
        prompt = PROMPTS[topic]
        (directory / 'prompt.txt').write_text(prompt + '\n')
        config = directory / 'capture.toml'
        config.write_text(setting(A.read_text(), 'log_path', json.dumps(str(directory / 'capture.engine.log'))))
        run_env = dict(env, ZEROMOE_PROMPT_TOKEN_FILE=str(directory / 'prompt.tokens'),
            ZEROMOE_GENERATED_TOKEN_FILE=str(directory / 'continuation.tokens'),
            ZEROMOE_QWEN_EARLY_READY_ROUTE_TRACE=str(directory / 'capture.routes.jsonl'))
        if args.resume and (directory / 'prediction_scores.json').exists():
            scores = json.loads((directory / 'prediction_scores.json').read_text())
            print(f'Reusing captured {topic}', flush=True)
        else:
            wait_idle(args.wait_idle)
            print(f'Capturing {topic}', flush=True)
            with (directory / 'capture.stdout').open('w') as output:
                subprocess.run([str(engine), '--graph-device-expert-layers', '0', '--graph-device-expert-trailing', '1', '--config', str(config), '--predict', str(args.ngen)],
                    input=prompt + '\n:q\n', text=True, cwd=ROOT, env=run_env,
                    stdout=output, stderr=subprocess.STDOUT, check=True)
            scores = analyze(directory / 'capture.routes.jsonl',
                ROOT / 'tests/moe_usage_survey/zeromoe_usage_survey_qwen4exp_usage_tap', 12.3e9)
            (directory / 'prediction_scores.json').write_text(json.dumps(scores, indent=2) + '\n')
        command = [sys.executable, str(ROOT / 'bench/option_b_early_ready_check.py'),
            '--engine', str(engine), '--out-dir', str(directory / 'comparison'),
            '--prompt-ids', str(directory / 'prompt.tokens'),
            '--continuation', str(directory / 'continuation.tokens'), '--reps', '1',
            '--all-split', '--dump-logits', '--allow-busy-correctness', '--wait-idle', str(args.wait_idle),
            '--baseline-env', 'ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT=1',
            '--baseline-env', 'ZEROMOE_QWEN_EARLY_READY_RESIDENT_TABLES=1',
            '--baseline-env', 'ZEROMOE_QWEN_EARLY_READY_PREDICT=1',
            '--candidate-env', 'ZEROMOE_QWEN_EARLY_READY_RESIDENT_TABLES=1',
            '--candidate-env', 'ZEROMOE_QWEN_EARLY_READY_PREDICT=1',
            '--candidate-env', f'ZEROMOE_QWEN_EARLY_READY_ROUTE_TRACE={directory / "staging.routes.jsonl"}',
            '--candidate-env', f'ZEROMOE_QWEN_EARLY_READY_STAGE_LAYER={args.stage_layer}']
        if args.verify:
            command += ['--candidate-env', 'ZEROMOE_QWEN_EARLY_READY_STAGE_VERIFY=1']
        if args.free_running:
            command += ['--free-running']
        if args.resume and (directory / 'comparison/manifest.json').exists():
            command += ['--resume']
        with (directory / 'comparison.log').open('a' if args.resume else 'w') as output:
            subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)
        staged_scores = analyze(directory / 'staging.routes.jsonl',
            ROOT / 'tests/moe_usage_survey/zeromoe_usage_survey_qwen4exp_usage_tap', 12.3e9)
        (directory / 'staging_scores.json').write_text(json.dumps(staged_scores, indent=2) + '\n')
        print(f'Completed {topic}; cold recall={scores["summary"]["proxy"]["cold_recall"]:.3%}', flush=True)


if __name__ == '__main__':
    main()
