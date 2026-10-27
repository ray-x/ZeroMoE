#!/usr/bin/env python3
"""Order-balanced Q2_0 / UD-IQ1_M comparison with a shared config and prompt.

Captures one continuation per quant and replays it for every measured repeat.
Raw text is retained in the warmup stdout; teacher-forced text is not a quality
test. Reports TOKEN-LOOP-DIAG steps >= 3, excluding prefill and decode setup.
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

from option_b_global_pool_check import ROOT, A, setting

MODELS = {
    "Q2": os.environ.get("ZEROMOE_QWEN_Q2_MODEL"),
    "IQ1": os.environ.get("ZEROMOE_QWEN_IQ1_MODEL"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--ngen", type=int, default=64)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS),
                   help="checkpoint(s) to include")
    p.add_argument("--teacher-force", action="store_true")
    p.add_argument("--bf16-cublas", action="store_true",
                   help="enable the GGML BF16 MMVF -> cuBLAS experiment")
    p.add_argument("--prompt-ids", type=Path,
                   help="reuse a saved prompt tokenization across binaries")
    p.add_argument("--continuation", type=Path,
                   help="force this fixed continuation on every measured run")
    p.add_argument("--dump-logits", action="store_true",
                   help="save raw per-step logits; increases file I/O")
    p.add_argument("--history", choices=["own", "Q2", "IQ1"], default="own",
                   help="teacher-forced history: one per quant, or a shared quant's continuation")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prompt", default="Explain briefly the key concepts of neural networks and gradient descent in plain English.")
    args = p.parse_args()
    if args.reps < 1 or args.ngen < 4:
        p.error("need positive reps and at least 4 generated tokens")
    missing_models = [name for name in args.models if not MODELS[name]]
    if missing_models:
        p.error("set " + " and ".join(
            f"ZEROMOE_QWEN_{name}_MODEL" for name in missing_models
        ))
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    engine = args.engine.resolve()
    engine_hash = hashlib.sha256(engine.read_bytes()).hexdigest()
    manifest = {
        "engine": str(engine), "engine_sha256": engine_hash,
        "threads": args.threads, "reps": args.reps, "ngen": args.ngen,
        "models_to_run": args.models,
        "teacher_force": args.teacher_force, "bf16_cublas": args.bf16_cublas,
        "dump_logits": args.dump_logits,
        "history": args.history, "prompt": args.prompt,
        "models": MODELS,
        "prompt_ids_sha256": hashlib.sha256(args.prompt_ids.read_bytes()).hexdigest() if args.prompt_ids else None,
        "continuation_sha256": hashlib.sha256(args.continuation.read_bytes()).hexdigest() if args.continuation else None,
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("resume settings or engine differ from saved run")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ZEROMOE_")}
    for key in ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OMP_PROC_BIND", "OMP_PLACES"):
        env.pop(key, None)
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS": "0",
        "ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING": "1",
        "ZEROMOE_QWEN_CPU_THREADS": str(args.threads),
        "ZEROMOE_BENCH_IGNORE_STOP": "1",
    })
    env["GGML_CUDA_BF16_MMVF_CUBLAS"] = "1" if args.bf16_cublas else "0"
    base = A.read_text()
    prompt_ids = out / "prompt.tokens"
    if args.prompt_ids:
        prompt_ids.write_bytes(args.prompt_ids.read_bytes())
    rows = []

    def run(name, rep):
        if hashlib.sha256(engine.read_bytes()).hexdigest() != engine_hash:
            raise RuntimeError("engine changed during measurement")
        tag = f"{name}_{rep}"
        log = out / f"{tag}.engine.log"
        reuse = args.resume and log.exists() and (out / f"{tag}.stdout").exists()
        while not reuse:
            gpu_procs = subprocess.check_output([
                "nvidia-smi", "--id=0", "--query-compute-apps=pid,process_name",
                "--format=csv,noheader"], text=True).strip()
            competing = subprocess.run(["pgrep", "-x", "-a", "zeromoe"], capture_output=True, text=True).stdout.strip()
            if not gpu_procs and not competing:
                break
            print(f"Waiting for competing inference: {gpu_procs or competing}", flush=True)
            time.sleep(5)
        config = setting(base, "model_path", json.dumps(MODELS[name]))
        config = setting(config, "log_path", json.dumps(str(log)))
        config = setting(config, "host_ram_gb", "90")
        cfg = out / f"{tag}.toml"
        cfg.write_text(config)
        runtime = env.copy()
        if args.prompt_ids:
            runtime["ZEROMOE_PROMPT_TOKEN_INPUT_FILE"] = str(prompt_ids)
        elif not prompt_ids.exists():
            runtime["ZEROMOE_PROMPT_TOKEN_FILE"] = str(prompt_ids)
        else:
            runtime["ZEROMOE_PROMPT_TOKEN_INPUT_FILE"] = str(prompt_ids)
        if rep == "warm":
            runtime["ZEROMOE_GENERATED_TOKEN_FILE"] = str(out / f"{name}.tokens")
        elif args.teacher_force:
            if args.continuation:
                runtime["ZEROMOE_TEACHER_FORCE_TOKEN_FILE"] = str(args.continuation.resolve())
            else:
                history = name if args.history == "own" else args.history
                runtime["ZEROMOE_TEACHER_FORCE_TOKEN_FILE"] = str(out / f"{history}.tokens")
            # Keep sampling enabled: the normal sampler cost remains in both arms.
        if args.dump_logits and rep != "warm":
            runtime["ZEROMOE_DECODE_LOGITS_DUMP"] = str(out / f"{tag}.logits")
        print(f"{'Reusing' if reuse else 'Running'} {tag}...", flush=True)
        started = time.monotonic()
        if not reuse:
            with (out / f"{tag}.stdout").open("w") as stdout:
                subprocess.run([str(engine), "--verbosity", "5", "--config", str(cfg), "--predict", str(args.ngen)],
                               input=args.prompt + "\n:q\n", text=True, cwd=ROOT, env=runtime,
                               stdout=stdout, stderr=subprocess.STDOUT, check=True)
        contents = log.read_text()
        if rep == "warm":
            text = (out / f"{tag}.stdout").read_text().split("Output:", 1)[-1]
            if any(marker in text for marker in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>")):
                raise RuntimeError(f"{tag}: continuation passed a stop token; use a longer prompt")
        timing = [(int(step), float(total), float(core)) for step, total, core in re.findall(
            r'TOKEN-LOOP-DIAG.*step=(\d+) iter_total=([\d.]+)ms.*core\(hc\+forward\)=([\d.]+)ms', contents)]
        steady = [(total, core) for step, total, core in timing if step >= 3]
        if len(steady) != args.ngen - 3:
            raise RuntimeError(f"{tag}: incomplete timing ({len(steady)} steps)")
        if re.search(r'using layered decode|CUDA error', contents):
            raise RuntimeError(f"{tag}: graph fallback or CUDA failure; inspect log")
        totals, cores = zip(*steady)
        row = {"model": name, "rep": rep, "decode_tps": 1000 / statistics.mean(totals),
               "core_ms": statistics.mean(cores), "loop_ms": statistics.mean(totals),
               "wall_s": None if reuse else time.monotonic() - started}
        print(json.dumps(row), flush=True)
        if rep != "warm":
            rows.append(row)
            (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")

    for name in args.models:
        run(name, "warm")
    for rep in range(1, args.reps + 1):
        for name in (("Q2", "IQ1") if rep % 2 else ("IQ1", "Q2")):
            if name not in args.models:
                continue
            run(name, rep)
    summary = {}
    for name in args.models:
        rates = [r["decode_tps"] for r in rows if r["model"] == name]
        summary[name] = {"mean_tps": statistics.mean(rates),
                         "sd_tps": statistics.stdev(rates) if len(rates) > 1 else 0,
                         "min_tps": min(rates), "max_tps": max(rates)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
