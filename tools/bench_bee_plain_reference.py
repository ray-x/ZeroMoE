#!/usr/bin/env python3
"""Matched plain-decode benchmark for the Bee Qwen4Exp Q2_0 MTP reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EVAL_RE = re.compile(r"eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s+(?:runs|tokens)\s*\([^)]*?([0-9.]+)\s+tokens per second", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def cmake_values(cache: Path) -> dict[str, str]:
    result = {}
    if cache.is_file():
        for line in cache.read_text(errors="replace").splitlines():
            if not line or line.startswith(("#", "//")) or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.startswith(("CMAKE_BUILD_TYPE:", "CMAKE_CUDA_ARCHITECTURES:",
                               "CMAKE_CUDA_COMPILER:", "GGML_CUDA:",
                               "GGML_CUDA_GRAPHS:", "GGML_CUDA_KVARN:")):
                result[key.split(":", 1)[0]] = value
    return result


def gpu_state(gpu: str) -> tuple[bool, str]:
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader", f"--id={gpu}"], capture_output=True, text=True)
    if query.returncode == 0:
        rows = [x.strip() for x in query.stdout.splitlines() if x.strip()]
        if rows:
            return True, "\n".join(rows)
    else:
        # NVML can fail transiently while CUDA still works. fuser provides a
        # conservative fallback: any process holding an NVIDIA device is busy.
        fuser = shutil.which("fuser")
        if not fuser:
            return True, f"nvidia-smi failed and fuser is unavailable: {query.stderr.strip()}"
        device = f"/dev/nvidia{gpu}"
        fallback = subprocess.run(
            [fuser, device, "/dev/nvidiactl", "/dev/nvidia-uvm"],
            capture_output=True, text=True)
        if fallback.returncode == 0:
            return True, f"nvidia-smi failed; GPU device holders: {fallback.stdout} {fallback.stderr}"
    for name in ("zeromoe", "llama-cli", "llama-speculative-simple", "nsys"):
        p = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True)
        if p.stdout.strip():
            return True, f"{name} pid={p.stdout.strip()}"
    return False, "idle (nvidia-smi or fuser fallback)"


def wait_for_idle(gpu: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while True:
        busy, detail = gpu_state(gpu)
        if not busy:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU busy; refusing to overlap another run:\n{detail}")
        time.sleep(5)


def parse_log(path: Path, expected_tokens: int) -> dict:
    text = path.read_text(errors="replace")
    matches = [m for line in text.splitlines()
               if "eval time" in line and "prompt eval time" not in line
               for m in EVAL_RE.finditer(line)]
    if not matches:
        raise RuntimeError(f"{path}: missing decode-only llama_print_timings eval time")
    elapsed_ms, count, tps = matches[-1].groups()
    count = int(count)
    if count != expected_tokens:
        raise RuntimeError(f"{path}: Bee plain decoded {count}, expected {expected_tokens}")
    return {"decoded_tokens": count, "decode_seconds": float(elapsed_ms) / 1000.0,
            "decode_tps": float(tps), "raw_log": str(path)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bee-binary", type=Path, required=True)
    p.add_argument("--bee-repo", type=Path, required=True)
    p.add_argument("--bee-build-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="main Q2_0 GGUF shard 1")
    p.add_argument("--prompt-dir", type=Path, default=ROOT / "bench/prompts/mtp_suite")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--expected-prompts", type=int, default=6)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--context", type=int, default=1536)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--gpu", default="0")
    p.add_argument("--cpu-moe-first", type=int, default=22)
    p.add_argument("--gpu-layers", default="99")
    p.add_argument("--cuda-delay-ms", type=int, default=500)
    p.add_argument("--idle-timeout", type=int, default=120)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    try:
        binary, repo = args.bee_binary.resolve(), args.bee_repo.resolve()
        build, model = args.bee_build_dir.resolve(), args.model.resolve()
        prompt_dir, out_dir = args.prompt_dir.resolve(), args.out_dir.resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"Bee CLI not executable: {binary}")
        if not model.is_file() or not prompt_dir.is_dir():
            raise ValueError("missing main model shard or prompt directory")
        prompts = sorted(x for x in prompt_dir.iterdir() if x.is_file() and not x.name.startswith("."))
        if args.expected_prompts and len(prompts) != args.expected_prompts:
            raise ValueError(f"expected {args.expected_prompts} prompts, found {len(prompts)}")
        texts = {x.name: x.read_text().strip() for x in prompts}
        if any(not x or "\n" in x for x in texts.values()):
            raise ValueError("prompt files must be nonempty single lines")
        if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
            raise ValueError(f"output directory is not empty: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        delay_c = out_dir / "cuda_startup_delay.c"
        delay_so = out_dir / "cuda_startup_delay.so"
        delay_c.write_text("#include <time.h>\n__attribute__((constructor)) static void delay_cuda_init(void) {\n"
                           f"  struct timespec ts = {{0, {args.cuda_delay_ms * 1000000}}};\n  nanosleep(&ts, 0);\n}}\n")
        cc = shutil.which("cc")
        if not cc:
            raise RuntimeError("cc is required for the CUDA startup-delay preload")
        subprocess.run([cc, "-std=gnu11", "-O2", "-shared", "-fPIC", "-o", str(delay_so), str(delay_c)], check=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
        env.update({"CUDA_VISIBLE_DEVICES": args.gpu, "OMP_NUM_THREADS": str(args.threads),
                    "LD_PRELOAD": str(delay_so) + ((":" + env["LD_PRELOAD"]) if env.get("LD_PRELOAD") else "")})
        shards = sorted(model.parent.glob("*-of-00002.gguf"))
        manifest = {
            "mode": "plain", "bee_binary": str(binary), "bee_binary_sha256": sha256(binary),
            "bee_source_commit": git(repo, "rev-parse", "HEAD"),
            "bee_source_dirty_status": git(repo, "status", "--short"),
            "bee_build_dir": str(build), "bee_cmake": cmake_values(build / "CMakeCache.txt"),
            "main_model_shards": [{"path": str(x), "bytes": x.stat().st_size,
                                   "mtime_ns": x.stat().st_mtime_ns} for x in shards],
            "prompt_dir": str(prompt_dir), "prompt_hashes": {k: hashlib.sha256(v.encode()).hexdigest() for k,v in texts.items()},
            "tokens": args.tokens, "context": args.context, "threads": args.threads,
            "gpu": args.gpu, "cpu_moe_first_layers": args.cpu_moe_first,
            "gpu_layers": args.gpu_layers, "temperature": 0.0, "top_k": 1,
            "timing_boundary": "llama_print_timings eval time (decode only; excludes load and prompt eval)",
        }
        mp = out_dir / "manifest.json"
        if args.resume and mp.is_file():
            if json.loads(mp.read_text()) != manifest:
                raise ValueError("resume manifest does not match Bee/model/prompt/flags")
        else:
            mp.write_text(json.dumps(manifest, indent=2) + "\n")
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        (out_dir / "nvidia-smi.txt").write_text(smi.stdout or smi.stderr)
        if not args.dry_run:
            wait_for_idle(args.gpu, args.idle_timeout)
            dev = subprocess.run([str(binary), "--list-devices"], cwd=repo, env=env,
                                 text=True, capture_output=True, timeout=120)
            (out_dir / "available-devices.txt").write_text(dev.stdout + "\n" + dev.stderr)
            if dev.returncode or not re.search(r"CUDA0:.*RTX 3090", dev.stdout + dev.stderr, re.I):
                raise RuntimeError("Bee CLI preflight did not find the RTX 3090")
        results = {}
        for rep in range(args.repeats):
            order = prompts if rep % 2 == 0 else list(reversed(prompts))
            for prompt in order:
                name = prompt.stem
                rd = out_dir / name / f"rep-{rep+1:02d}"
                result_file = rd / "result.json"
                if args.resume and result_file.is_file():
                    results.setdefault(prompt.name, []).append(json.loads(result_file.read_text()))
                    continue
                if rd.exists():
                    shutil.rmtree(rd)
                rd.mkdir(parents=True)
                cmd = [str(binary), "-m", str(model), "-p", texts[prompt.name],
                       "-n", str(args.tokens), "-c", str(args.context), "-t", str(args.threads),
                       "-ngl", str(args.gpu_layers), "-ncmoe", str(args.cpu_moe_first),
                       "--temp", "0", "--top-k", "1", "--seed", "42", "--ignore-eos",
                       "--single-turn", "--color", "off", "-lv", "4"]
                (rd / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
                if args.dry_run:
                    print(f"DRY RUN {prompt.name} rep={rep+1}: {cmd}")
                    continue
                wait_for_idle(args.gpu, args.idle_timeout)
                print(f"{prompt.name} rep={rep+1} Bee plain", flush=True)
                run_log = rd / "run.log"
                with run_log.open("w") as stream:
                    proc = subprocess.run(cmd, cwd=repo, env=env, text=True,
                                          stdout=stream, stderr=subprocess.STDOUT,
                                          timeout=args.timeout)
                if proc.returncode:
                    raise RuntimeError(f"Bee plain failed rc={proc.returncode}; see {run_log}")
                result = parse_log(run_log, args.tokens)
                result.update({"prompt": prompt.name, "rep": rep+1, "command": cmd})
                result_file.write_text(json.dumps(result, indent=2) + "\n")
                results.setdefault(prompt.name, []).append(result)
        if args.dry_run:
            print(f"dry-run: commands written under {out_dir}; no GPU process launched")
            return 0
        summaries = {}
        for name, runs in results.items():
            summaries[name] = {"runs": len(runs),
                               "median_decode_tps": statistics.median(x["decode_tps"] for x in runs),
                               "median_decode_seconds": statistics.median(x["decode_seconds"] for x in runs)}
        (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(json.dumps(summaries, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
