#!/usr/bin/env python3
"""Teacher-force one continuation through two Qwen4Exp placement configs.

The baseline generates a fixed token stream. The candidate is forced through
that same stream, and raw per-step logits are compared. MTP is disabled in both
configs: this gate measures placement quality without speculative decoding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from compare_decode_logits import metrics, records  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_root_key(text: str, key: str, value: str) -> str:
    pattern = rf"(?m)^{re.escape(key)}\s*=.*$"
    if re.search(pattern, text):
        return re.sub(pattern, f"{key} = {value}", text)
    # Root keys belong before the first TOML table.
    split = re.search(r"(?m)^\s*\[\[", text)
    at = split.start() if split else len(text)
    return text[:at] + f"{key} = {value}\n" + text[at:]


def parse_override(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"override must be KEY=TOML_VALUE: {value!r}")
    key, toml_value = value.split("=", 1)
    key = key.strip()
    if not re.fullmatch(r"[A-Za-z0-9_]+", key):
        raise ValueError(f"invalid TOML root key: {key!r}")
    if not toml_value.strip():
        raise ValueError(f"missing TOML value for {key}")
    return key, toml_value.strip()


def parse_env_overrides(values: list[str]) -> dict[str, str]:
    reserved = {
        "CUDA_VISIBLE_DEVICES", "ZEROMOE_DECODE_LOGITS_DUMP",
        "ZEROMOE_GENERATED_TOKEN_FILE", "ZEROMOE_TEACHER_FORCE_TOKEN_FILE",
        "ZEROMOE_TEACHER_FORCE_SKIP_SAMPLING", "ZEROMOE_BENCH_IGNORE_STOP",
        "ZEROMOE_QWEN4EXP_FULL_GRAPH", "ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE",
    }
    result = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"environment override must be KEY=VALUE: {item!r}")
        key, value = item.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid environment key: {key!r}")
        if key in reserved:
            raise ValueError(f"environment key is controlled by the gate: {key}")
        result[key] = value
    return result


def make_config(base: str, name: str, log_path: Path, tokens: int,
                top_k: int | None, usage_file: Path | None,
                prefill_usage_file: Path | None,
                overrides: list[str]) -> str:
    # This gate is about expert placement. Strip all MTP keys to prevent a
    # config copied from an MTP run from silently entering a different path.
    lines = [line for line in base.splitlines()
             if not re.match(r"^\s*mtp_[A-Za-z0-9_]+\s*=", line)]
    text = "\n".join(lines) + "\n"
    text = set_root_key(text, "log_path", json.dumps(str(log_path.resolve())))
    text = set_root_key(text, "max_new_tokens", str(tokens))
    text = set_root_key(text, "temperature", "0.0")
    text = set_root_key(text, "freeze_usage_stats", "true")
    if top_k is not None:
        text = set_root_key(text, "qwen4exp_option_b_global_top_k", str(top_k))
    if usage_file is not None:
        text = set_root_key(text, "pin_file", json.dumps(str(usage_file.resolve())))
    if prefill_usage_file is not None:
        text = set_root_key(text, "prefill_pin_file",
                            json.dumps(str(prefill_usage_file.resolve())))
    for override in overrides:
        key, value = parse_override(override)
        if key.startswith("mtp_"):
            raise ValueError("MTP overrides are not allowed in the placement accuracy gate")
        text = set_root_key(text, key, value)
    if re.search(r"(?m)^\s*mtp_[A-Za-z0-9_]+\s*=", text):
        raise RuntimeError(f"{name}: MTP config keys remain after sanitizing")
    return text


def wait_for_idle(gpu: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    warned = False
    while True:
        query = subprocess.run(
            ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"], capture_output=True, text=True)
        if query.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed before run: {query.stderr.strip()}")
        engines = subprocess.run(["pgrep", "-x", "zeromoe"],
                                 capture_output=True, text=True).stdout.strip()
        if not query.stdout.strip() and not engines:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU busy; refusing to overlap inference: "
                               f"{query.stdout.strip() or engines}")
        if not warned:
            print(f"Waiting for idle GPU {gpu}: {query.stdout.strip() or engines}",
                  flush=True)
            warned = True
        time.sleep(5)


def run_engine(engine: Path, engine_repo: Path, config: Path, prompt: str,
               env: dict[str, str], stdout_path: Path, timeout: int) -> None:
    with stdout_path.open("w") as output:
        subprocess.run([str(engine), "--graph-device-expert-layers", "0",
                            "--graph-device-expert-trailing", "1",
                            "--config", str(config)], input=prompt + "\n:q\n",
                       text=True, stdout=output, stderr=subprocess.STDOUT,
                       cwd=engine_repo, env=env, timeout=timeout, check=True)


def compare_dumps(reference_path: Path, candidate_path: Path,
                  token_ids: list[int], expected_steps: int) -> dict:
    a_records, b_records = records(reference_path), records(candidate_path)
    per_token = []
    while True:
        a, b = next(a_records, None), next(b_records, None)
        if a is None or b is None:
            if a is not None or b is not None:
                raise ValueError("baseline and candidate logit record counts differ")
            break
        step, a_logits = a
        if step != b[0] or step >= len(token_ids):
            raise ValueError(f"logit/token step mismatch: {step} vs {b[0]}")
        row = metrics(a_logits, b[1], token_ids[step])
        dot = math.fsum(x * y for x, y in zip(a_logits, b[1]))
        na = math.sqrt(math.fsum(x * x for x in a_logits))
        nb = math.sqrt(math.fsum(y * y for y in b[1]))
        row["logit_cosine"] = dot / (na * nb) if na and nb else float(na == nb)
        row["step"] = step
        per_token.append(row)
    if len(per_token) != expected_steps:
        raise ValueError(f"expected {expected_steps} logit records, found {len(per_token)}")
    summary = {
        "steps": len(per_token),
        "top1_agreement": sum(r["top1_agrees"] for r in per_token) / len(per_token),
        "mean_logit_cosine": math.fsum(r["logit_cosine"] for r in per_token) / len(per_token),
        "min_logit_cosine": min(r["logit_cosine"] for r in per_token),
        "mean_kl_ref_candidate": math.fsum(r["kl_ref_candidate"] for r in per_token) / len(per_token),
        "max_kl_ref_candidate": max(r["kl_ref_candidate"] for r in per_token),
        "mean_teacher_nll_delta": math.fsum(r["teacher_nll_delta"] for r in per_token) / len(per_token),
        "first_top1_disagreement": next((r for r in per_token if not r["top1_agrees"]), None),
    }
    return {"summary": summary, "per_token": per_token}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=ROOT / "build/zeromoe")
    parser.add_argument("--engine-repo", type=Path, default=ROOT,
                        help="source worktree that built --engine (supports a wt-swap-b candidate binary)")
    parser.add_argument("--config", type=Path, required=True,
                        help="working Qwen4Exp full-graph TOML")
    parser.add_argument("--prompt-dir", type=Path, default=ROOT / "scratchpad/hy/tr")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--expected-prompts", type=int, default=6,
                        help="expected prompt files; use 0 to disable the count check")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--baseline-name", default="static-13000")
    parser.add_argument("--candidate-name", default="static-11000")
    parser.add_argument("--baseline-top-k", type=int, default=13000)
    parser.add_argument("--candidate-top-k", type=int, default=11000)
    parser.add_argument("--baseline-usage-file", type=Path)
    parser.add_argument("--candidate-usage-file", type=Path)
    parser.add_argument("--prefill-usage-file", type=Path)
    parser.add_argument("--baseline-set", action="append", default=[], metavar="KEY=VALUE",
                        help="additional baseline root TOML override")
    parser.add_argument("--candidate-set", action="append", default=[], metavar="KEY=VALUE",
                        help="additional candidate root TOML override")
    parser.add_argument("--baseline-env", action="append", default=[], metavar="KEY=VALUE",
                        help="additional baseline-only runtime environment override")
    parser.add_argument("--candidate-env", action="append", default=[], metavar="KEY=VALUE",
                        help="additional candidate-only runtime environment override")
    parser.add_argument("--idle-timeout", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--resume", action="store_true",
                        help="reuse completed prompt summaries in an existing matching output directory")
    args = parser.parse_args()

    try:
        engine = args.engine.resolve()
        engine_repo = args.engine_repo.resolve()
        config = args.config.resolve()
        prompt_dir = args.prompt_dir.resolve()
        out_dir = args.out_dir.resolve()
        if not engine.is_file() or not os.access(engine, os.X_OK):
            raise ValueError(f"engine is not executable: {engine}")
        if not config.is_file():
            raise ValueError(f"missing config: {config}")
        if not prompt_dir.is_dir():
            raise ValueError(f"missing prompt directory: {prompt_dir}")
        if out_dir.exists() and not args.resume:
            raise ValueError(f"output directory already exists: {out_dir}")
        if args.tokens <= 0 or args.threads <= 0:
            raise ValueError("tokens and threads must be positive")
        prompts = sorted(p for p in prompt_dir.iterdir() if p.is_file() and not p.name.startswith("."))
        if args.expected_prompts and len(prompts) != args.expected_prompts:
            raise ValueError(f"expected {args.expected_prompts} prompt files, found {len(prompts)}")
        if not prompts:
            raise ValueError(f"no prompt files in {prompt_dir}")
        prompt_text = {}
        for path in prompts:
            text = path.read_text().strip()
            if not text or "\n" in text:
                raise ValueError(f"prompt must be one nonempty line: {path}")
            prompt_text[path.name] = text

        subprocess.run([sys.executable, str(ROOT / "tools/build_provenance.py"),
                        "check", "--binary", str(engine), "--repo", str(engine_repo)],
                       cwd=ROOT, check=True)
        base = config.read_text()
        manifest = {
            "worktree_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "engine_repo": str(engine_repo),
            "engine_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=engine_repo, text=True).strip(),
            "engine": str(engine), "engine_sha256": sha256(engine),
            "build_info": str(Path(str(engine) + ".build-info.json")),
            "base_config": str(config), "base_config_sha256": sha256(config),
            "prompt_dir": str(prompt_dir),
            "prompts": {p.name: hashlib.sha256(prompt_text[p.name].encode()).hexdigest() for p in prompts},
            "tokens": args.tokens, "context": args.context, "threads": args.threads,
            "gpu": args.gpu, "baseline": args.baseline_name, "candidate": args.candidate_name,
            "baseline_top_k": args.baseline_top_k, "candidate_top_k": args.candidate_top_k,
            "baseline_set": args.baseline_set, "candidate_set": args.candidate_set,
            "baseline_env": args.baseline_env, "candidate_env": args.candidate_env,
        }
        if out_dir.exists():
            manifest_path = out_dir / "manifest.json"
            if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("resume manifest does not match this engine/config/prompt run")
        else:
            out_dir.mkdir(parents=True)
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (out_dir / "build-info.txt").write_text(Path(str(engine) + ".build-info.txt").read_text())

        results = {}
        common_env = {k: v for k, v in os.environ.items() if not k.startswith("ZEROMOE_")}
        common_env.update({
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "OMP_NUM_THREADS": str(args.threads),
            "ZEROMOE_QWEN4EXP_FULL_GRAPH": "1",
            "ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE": "1",
            "ZEROMOE_QWEN_CPU_THREADS": str(args.threads),
            "ZEROMOE_BENCH_IGNORE_STOP": "1",
        })
        for prompt_path in prompts:
            key = re.sub(r"[^A-Za-z0-9_.-]+", "_", prompt_path.stem)
            prompt = prompt_text[prompt_path.name]
            prompt_dir_out = out_dir / key
            summary_path = prompt_dir_out / "summary.json"
            if args.resume and summary_path.is_file():
                results[prompt_path.name] = json.loads(summary_path.read_text())
                print(f"{prompt_path.name}: resume skipped completed prompt", flush=True)
                continue
            if prompt_dir_out.exists():
                if not args.resume:
                    raise ValueError(f"prompt output already exists: {prompt_dir_out}")
                shutil.rmtree(prompt_dir_out)
            prompt_dir_out.mkdir()
            token_file = prompt_dir_out / "baseline.tokens"
            cell_summary = {}
            for name, topk, usage, overrides, env_overrides, forced in (
                (args.baseline_name, args.baseline_top_k, args.baseline_usage_file,
                 args.baseline_set, args.baseline_env, False),
                (args.candidate_name, args.candidate_top_k, args.candidate_usage_file,
                 args.candidate_set, args.candidate_env, True),
            ):
                run_dir = prompt_dir_out / name
                run_dir.mkdir()
                logits_path = run_dir / "decode.logits"
                generated_path = run_dir / "generated.tokens"
                log_path = run_dir / "engine.log"
                toml_path = run_dir / "config.toml"
                if args.context is not None:
                    overrides = [*overrides, f"context_size={args.context}"]
                cfg_text = make_config(base, name, log_path, args.tokens, topk, usage,
                                       args.prefill_usage_file, overrides)
                toml_path.write_text(cfg_text)
                env = common_env.copy()
                env["ZEROMOE_DECODE_LOGITS_DUMP"] = str(logits_path)
                env["ZEROMOE_GENERATED_TOKEN_FILE"] = str(generated_path)
                if forced:
                    env["ZEROMOE_TEACHER_FORCE_TOKEN_FILE"] = str(token_file)
                    env["ZEROMOE_TEACHER_FORCE_SKIP_SAMPLING"] = "1"
                env.update(parse_env_overrides(env_overrides))
                wait_for_idle(args.gpu, args.idle_timeout)
                print(f"{prompt_path.name}: run {name} ({args.tokens} tokens)", flush=True)
                run_engine(engine, engine_repo, toml_path, prompt, env,
                           run_dir / "stdout.txt", args.timeout)
                if sum(1 for _ in records(logits_path)) != args.tokens:
                    raise RuntimeError(f"{prompt_path.name}/{name}: expected {args.tokens} logit rows")
                generated = [int(x) for x in generated_path.read_text().split()]
                if len(generated) != args.tokens:
                    raise RuntimeError(f"{prompt_path.name}/{name}: expected {args.tokens} generated/forced tokens, got {len(generated)}")
                if not forced:
                    token_file.write_text("".join(f"{x}\n" for x in generated))
                    cell_summary["baseline_tokens_sha256"] = sha256(token_file)
                else:
                    if generated != [int(x) for x in token_file.read_text().split()]:
                        raise RuntimeError(f"{prompt_path.name}: candidate token input changed")
                    result = compare_dumps(prompt_dir_out / args.baseline_name / "decode.logits",
                                           logits_path, generated, args.tokens)
                    (run_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
                    cell_summary["comparison"] = result["summary"]
                    print(f"{prompt_path.name}: top1={result['summary']['top1_agreement']:.2%} "
                          f"cos={result['summary']['mean_logit_cosine']:.6f} "
                          f"KL={result['summary']['mean_kl_ref_candidate']:.6g}", flush=True)
            results[prompt_path.name] = cell_summary
            summary_path.write_text(json.dumps(cell_summary, indent=2) + "\n")
        (out_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"Results written to {out_dir}")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
