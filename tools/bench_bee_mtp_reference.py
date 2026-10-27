#!/usr/bin/env python3
"""Run a repeatable llama.cpp/Bee Qwen4Exp MTP reference on a prompt suite.

The GGUF must already be spliced into three shards. This runner does not edit
the Bee checkout or model files; it records the binary/source identity, full
command, startup CUDA delay helper, raw logs, acceptance, tokens/window and
decode rate for every run.
"""
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
DECODE_RE = re.compile(
    r"decoded\s+(\d+)\s+tokens\s+in\s+([0-9.]+)\s+seconds,\s*"
    r"speed:\s*([0-9.]+)\s+t/s")
N_PREDICT_RE = re.compile(r"n_predict\s*=\s*(\d+)")
N_DRAFTED_RE = re.compile(r"n_drafted\s*=\s*(\d+)")
N_ACCEPT_RE = re.compile(r"n_accept\s*=\s*(\d+)")
ACCEPT_RE = re.compile(r"accept\s*=\s*([0-9.]+)%")
SPEC_STATS_RE = re.compile(
    r"statistics\s+draft-mtp:\s+#calls\(b,g,a\)\s*=\s*(\d+)\s+(\d+)\s+(\d+),\s*"
    r"#gen drafts\s*=\s*(\d+),\s*#acc drafts\s*=\s*(\d+),\s*"
    r"#gen tokens\s*=\s*(\d+),\s*#acc tokens\s*=\s*(\d+)([^\n]*)")
MEAN_LEN_RE = re.compile(r"#mean acc len\s*=\s*([0-9.]+)")


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
    if not cache.is_file():
        return result
    for line in cache.read_text(errors="replace").splitlines():
        if not line or line.startswith("//") or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.startswith(("CMAKE_BUILD_TYPE:", "CMAKE_CUDA_ARCHITECTURES:",
                           "CMAKE_CUDA_COMPILER:", "GGML_CUDA:",
                           "GGML_CUDA_GRAPHS:", "GGML_OPENMP:",
                           "GGML_BACKEND_DL:")):
            result[key.split(":", 1)[0]] = value
    return result


def write_cuda_delay(out_dir: Path, delay_ms: int) -> Path:
    source = out_dir / "cuda_startup_delay.c"
    library = out_dir / "cuda_startup_delay.so"
    source.write_text(
        "#include <time.h>\n"
        "__attribute__((constructor)) static void delay_cuda_init(void) {\n"
        f"  struct timespec ts = {{0, {delay_ms * 1000000}}};\n"
        "  nanosleep(&ts, 0);\n"
        "}\n")
    compiler = shutil.which("cc")
    if not compiler:
        raise RuntimeError("C compiler 'cc' is required to build the CUDA startup-delay preload")
    subprocess.run([compiler, "-std=gnu11", "-O2", "-shared", "-fPIC",
                    "-o", str(library), str(source)], check=True)
    return library


def gpu_state(gpu: str) -> tuple[bool, str]:
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader", f"--id={gpu}"],
        capture_output=True, text=True)
    if query.returncode != 0:
        return True, f"nvidia-smi query failed: {query.stderr.strip()}"
    rows = [x.strip() for x in query.stdout.splitlines() if x.strip()]
    for name in ("zeromoe", "llama-speculative-simple", "llama-cli"):
        p = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True)
        if p.stdout.strip():
            rows.extend(f"{name} pid={x}" for x in p.stdout.split())
    return bool(rows), "\n".join(rows)


def wait_for_idle(gpu: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while True:
        busy, detail = gpu_state(gpu)
        if not busy:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU busy; refusing overlap another inference:\n{detail}")
        time.sleep(5)


def parse_log(path: Path, expected_tokens: int) -> dict:
    text = path.read_text(errors="replace")
    decoded = list(DECODE_RE.finditer(text))
    if not decoded:
        raise RuntimeError(f"{path}: missing decoded-tokens/speed line")
    match = decoded[-1]
    n_decoded, seconds, tps = int(match.group(1)), float(match.group(2)), float(match.group(3))
    n_predict = list(N_PREDICT_RE.finditer(text))
    n_drafted = list(N_DRAFTED_RE.finditer(text))
    n_accept = list(N_ACCEPT_RE.finditer(text))
    accept_rate = list(ACCEPT_RE.finditer(text))
    stats = list(SPEC_STATS_RE.finditer(text))
    if n_decoded != expected_tokens:
        raise RuntimeError(f"{path}: Bee decoded {n_decoded}, expected {expected_tokens}")
    if not n_predict or not n_drafted or not n_accept or not accept_rate:
        raise RuntimeError(f"{path}: missing speculative acceptance counters")
    result = {
        "decoded_tokens": n_decoded,
        "decode_seconds": seconds,
        "decode_tps": tps,
        "n_predict": int(n_predict[-1].group(1)),
        "n_drafted_example": int(n_drafted[-1].group(1)),
        "n_accepted_example": int(n_accept[-1].group(1)),
        "acceptance_percent_example": float(accept_rate[-1].group(1)),
        "raw_log": str(path),
    }
    if stats:
        s = stats[-1]
        n_call_begin, n_call_draft, n_call_accept = map(int, s.groups()[:3])
        gen_drafts, acc_drafts, gen_tokens, acc_tokens = map(int, s.groups()[3:7])
        mean = MEAN_LEN_RE.search(s.group(8))
        result.update({
            "spec_calls_begin": n_call_begin,
            "spec_calls_draft": n_call_draft,
            "spec_windows": n_call_accept,
            "spec_generated_drafts": gen_drafts,
            "spec_accepted_drafts": acc_drafts,
            "spec_generated_tokens": gen_tokens,
            "spec_accepted_tokens": acc_tokens,
            "spec_window_acceptance_percent": 100.0 * acc_drafts / gen_drafts if gen_drafts else 0.0,
"spec_candidate_token_acceptance_percent": 100.0 * acc_tokens / gen_tokens if gen_tokens else 0.0,
            "mean_tokens_per_window": float(mean.group(1)) if mean else None,
            "window_metric_source": "common_speculative_print_stats",
        })
    else:
        # The example's aggregate counters include only committed drafts when
        # it reuses a partially accepted checkpoint. Keep that caveat explicit.
        cycles = max(1, result["n_predict"] - result["n_accepted_example"])
        result.update({
            "spec_windows": cycles,
            "mean_tokens_per_window": result["n_predict"] / cycles,
            "window_metric_source": "n_predict_minus_n_accept_example_counters",
        })
    return result


def summarize(runs: list[dict]) -> dict:
    result = {
        "runs": len(runs),
        "median_decode_tps": statistics.median(r["decode_tps"] for r in runs),
        "median_decode_seconds": statistics.median(r["decode_seconds"] for r in runs),
        "median_acceptance_percent": statistics.median(
            r.get("spec_candidate_token_acceptance_percent",
                  100.0 * r.get("spec_accepted_tokens", 0) / max(1, r.get("spec_generated_tokens", 0)))
            for r in runs),
        "median_window_acceptance_percent": statistics.median(
            r.get("spec_window_acceptance_percent", r.get("acceptance_percent_example", 0.0))
            for r in runs),
        "median_tokens_per_window": statistics.median(
            r["mean_tokens_per_window"] for r in runs if r.get("mean_tokens_per_window") is not None),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bee-binary", type=Path, required=True)
    parser.add_argument("--bee-repo", type=Path, required=True)
    parser.add_argument("--bee-build-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True,
                        help="corrected three-shard GGUF; pass shard 1")
    parser.add_argument("--prompt-dir", type=Path, default=ROOT / "bench/prompts/mtp_suite")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--n-predict-offset", type=int, default=1,
                    help="subtract from Bee -n; MTP stops after a whole verification window, then exact output count is checked")
    parser.add_argument("--expected-prompts", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--drafts", type=int, default=3)
    parser.add_argument("--draft-p-min", type=float, default=0.0,
                        help="minimum draft-token confidence; 0 disables the cutoff")
    parser.add_argument("--verbosity", type=int, default=4,
                        help="Bee log verbosity (5 includes debug draft-candidate IDs/probabilities)")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu-moe-first", type=int, default=30)
    parser.add_argument("--gpu-layers", default="99")
    parser.add_argument("--draft-gpu-layers", default="99")
    parser.add_argument("--cuda-delay-ms", type=int, default=500)
    parser.add_argument("--idle-timeout", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--allow-eos", action="store_true",
                        help="allow early EOS instead of forcing the fixed token budget")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.tokens - args.n_predict_offset <= 0:
        parser.error("--n-predict-offset must leave a positive Bee -n value")
    if not 0.0 <= args.draft_p_min <= 1.0:
        parser.error("--draft-p-min must be in [0, 1]")
    if not 0 <= args.verbosity <= 5:
        parser.error("--verbosity must be in [0, 5]")

    try:
        binary, bee_repo = args.bee_binary.resolve(), args.bee_repo.resolve()
        bee_build, model = args.bee_build_dir.resolve(), args.model.resolve()
        prompt_dir, out_dir = args.prompt_dir.resolve(), args.out_dir.resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"Bee binary not executable: {binary}")
        if not model.is_file() or not prompt_dir.is_dir():
            raise ValueError("missing spliced model shard 1 or prompt directory")
        prompts = sorted(p for p in prompt_dir.iterdir() if p.is_file() and not p.name.startswith("."))
        if args.expected_prompts and len(prompts) != args.expected_prompts:
            raise ValueError(f"expected {args.expected_prompts} prompt files, found {len(prompts)}")
        prompt_text = {p.name: p.read_text().strip() for p in prompts}
        if any(not text or "\n" in text for text in prompt_text.values()):
            raise ValueError("prompt files must be nonempty single lines")
        if args.tokens <= 0 or args.repeats <= 0 or args.drafts <= 0 or args.threads <= 0:
            raise ValueError("tokens, repeats, drafts, and threads must be positive")
        if out_dir.exists() and not args.resume:
            raise ValueError(f"output directory exists: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)

        delay_source = out_dir / "cuda_startup_delay.c"
        delay_library = out_dir / "cuda_startup_delay.so"
        delay_source.write_text(
            "#include <time.h>\n"
            "__attribute__((constructor)) static void delay_cuda_init(void) {\n"
            f"  struct timespec ts = {{0, {args.cuda_delay_ms * 1000000}}};\n"
            "  nanosleep(&ts, 0);\n}\n")
        compiler = shutil.which("cc")
        if not compiler:
            raise RuntimeError("cc is required to build the CUDA startup-delay preload")
        subprocess.run([compiler, "-std=gnu11", "-O2", "-shared", "-fPIC",
                        "-o", str(delay_library), str(delay_source)], check=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
        old_preload = env.get("LD_PRELOAD", "")
        env.update({"CUDA_VISIBLE_DEVICES": args.gpu,
                    "OMP_NUM_THREADS": str(args.threads),
                    "LD_PRELOAD": str(delay_library) + (":" + old_preload if old_preload else "")})

        stat = lambda path: {"path": str(path), "bytes": path.stat().st_size,
                             "mtime_ns": path.stat().st_mtime_ns}
        bee_status = git(bee_repo, "status", "--short")
        model_dir = model.parent
        splice_manifest = model_dir / "splice-manifest.json"
        manifest = {
            "bee_binary": str(binary), "bee_binary_sha256": sha256(binary),
            "bee_source_commit": git(bee_repo, "rev-parse", "HEAD"),
            "bee_source_dirty_status": bee_status,
            "bee_build_dir": str(bee_build), "bee_cmake": cmake_values(bee_build / "CMakeCache.txt"),
            "model_shard1": stat(model), "model_shards": [stat(p) for p in sorted(model_dir.glob("*-of-00003.gguf"))],
            "splice_manifest": str(splice_manifest) if splice_manifest.is_file() else None,
            "splice_manifest_sha256": sha256(splice_manifest) if splice_manifest.is_file() else None,
            "prompt_dir": str(prompt_dir),
            "prompt_hashes": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in prompt_text.items()},
            "tokens": args.tokens, "bee_cli_n_predict": args.tokens - args.n_predict_offset,
            "n_predict_offset": args.n_predict_offset, "context": args.context, "threads": args.threads,
            "gpu": args.gpu, "cpu_moe_first_layers": args.cpu_moe_first,
            "gpu_layers": args.gpu_layers, "draft_gpu_layers": args.draft_gpu_layers,
            "draft_tokens": args.drafts, "draft_p_min": args.draft_p_min,
            "verbosity": args.verbosity, "temperature": 0.0, "top_k": 1,
            "ignore_eos": not args.allow_eos, "cuda_startup_delay_ms": args.cuda_delay_ms,
        }
        manifest_path = out_dir / "manifest.json"
        if args.resume and manifest_path.is_file():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("resume manifest does not match Bee/model/prompt/flags")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (out_dir / "bee-source-status.txt").write_text(
            f"commit {manifest['bee_source_commit']}\n{bee_status}\n")
        if not args.dry_run:
            (out_dir / "nvidia-smi.txt").write_text(subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True).stdout)
            wait_for_idle(args.gpu, args.idle_timeout)
            device_env = env.copy()
            device_result = subprocess.run([str(binary), "--list-devices"],
                                           capture_output=True, text=True, env=device_env,
                                           cwd=bee_repo, timeout=120)
            device_text = device_result.stdout + "\n" + device_result.stderr
            (out_dir / "available-devices.txt").write_text(device_text)
            if device_result.returncode != 0 or not re.search(r"CUDA0:.*RTX 3090", device_text, re.I):
                raise RuntimeError(f"Bee CUDA preflight did not find RTX 3090; see {out_dir/'available-devices.txt'}")
        results = {}
        prompts_in_order = prompts
        for rep in range(args.repeats):
            ordered = prompts_in_order if rep % 2 == 0 else list(reversed(prompts_in_order))
            for prompt_file in ordered:
                key = re.sub(r"[^A-Za-z0-9_.-]+", "_", prompt_file.stem)
                run_dir = out_dir / key / f"rep-{rep+1:02d}"
                result_path = run_dir / "result.json"
                if args.resume and result_path.is_file():
                    result = json.loads(result_path.read_text())
                    results.setdefault(prompt_file.name, []).append(result)
                    continue
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                run_dir.mkdir(parents=True)
                command = [str(binary), "-m", str(model), "-p", prompt_text[prompt_file.name],
                           "--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.drafts),
                           "--spec-draft-p-min", str(args.draft_p_min),
                           "-ngld", str(args.draft_gpu_layers), "-ngl", str(args.gpu_layers),
                           "-ncmoe", str(args.cpu_moe_first), "-t", str(args.threads),
                           "-c", str(args.context), "-n", str(args.tokens - args.n_predict_offset), "--temp", "0",
                           "--top-k", "1", "--seed", "42", "--color", "off", "-lv", str(args.verbosity)]
                if not args.allow_eos:
                    command.append("--ignore-eos")
                (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                if args.dry_run:
                    print(f"DRY RUN {prompt_file.name} rep={rep+1}: {command}")
                    continue
                wait_for_idle(args.gpu, args.idle_timeout)
                print(f"{prompt_file.name} rep={rep+1} Bee MTP", flush=True)
                p = subprocess.run(command, cwd=bee_repo, env=env, text=True,
                                   capture_output=True, timeout=args.timeout)
                log = p.stdout + "\n" + p.stderr
                (run_dir / "run.log").write_text(log)
                if p.returncode:
                    raise RuntimeError(f"Bee MTP failed rc={p.returncode}; see {run_dir/'run.log'}")
                result = parse_log(run_dir / "run.log", args.tokens)
                result.update({"prompt": prompt_file.name, "rep": rep + 1,
                               "command": command, "returncode": p.returncode})
                result_path.write_text(json.dumps(result, indent=2) + "\n")
                results.setdefault(prompt_file.name, []).append(result)
            (out_dir / "progress.json").write_text(json.dumps(results, indent=2) + "\n")

        if args.dry_run:
            print(f"dry-run: commands written under {out_dir}; no GPU process launched")
            return 0
        summaries = {name: summarize(runs) for name, runs in results.items()}
        (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print("Bee MTP reference results:")
        for name, summary in summaries.items():
            print(f"{name}: {summary}")
        print(f"results: {out_dir}")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
