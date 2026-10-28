#!/usr/bin/env python3
"""Decode-only plain/MTP benchmark over a prompt directory.

Each prompt/mode is a fresh process. BENCH_IGNORE_STOP forces exactly N output
tokens for both modes. Per-run configs, stdout, engine logs and extracted timing
boundaries are kept under the output directory; summaries use medians.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WALL_RE = re.compile(r"\[REQUEST-DECODE-WALL\]\s+tokens=(\d+)\s+wall=([0-9.]+)s\s+rate=([0-9.]+)")
MTP_RE = re.compile(r"\[MTP-GRAPH\]\s+cycles=(\d+)\s+drafted=(\d+)\s+accepted=(\d+)\s+\(([0-9.]+)%\)\s+emitted=(\d+).*?target_steps=(\d+)\s+batches=(\d+)\s+batch_tokens=(\d+)\s+replay_tokens=(\d+)\s+draft_steps=(\d+)\s+restores=(\d+)")
PROFILE_RE = re.compile(r"\[MTP-PROFILE\]\s+speculate=([0-9.]+)\s*ms\s+windows=(\d+):.*?prepare=([0-9.]+)\s+run=([0-9.]+)\s+copy\+finite=([0-9.]+)\s+commit=([0-9.]+)\s+\|\s+draft steps=(\d+)\s+([0-9.]+)\s+ms\s+\|\s+other.*?=([0-9.]+)\s+ms")
BUILD_RE = re.compile(r"\[MTP-GRAPH-BUILD\].*?weights=(\d+)\s+state=(\d+).*?weight_upload_ms=([0-9.]+)\s+head_upload_ms=([0-9.]+)\s+build_ms=([0-9.]+)")
SETUP_RE = re.compile(r"\[MTP-GRAPH-SETUP\]\s+factory_ms=([0-9.]+)\s+runtime_ms=([0-9.]+)\s+measure_build_ms=([0-9.]+)\s+execution_build_ms=([0-9.]+)\s+scheduler_alloc_ms=([0-9.]+)\s+reset_ms=([0-9.]+)\s+weights=(\d+)\s+state=(\d+)\s+device_workspace=(\d+)\s+host_workspace=(\d+)\s+post_prefill_wait_ms=([0-9.]+)\s+hidden_ms=([0-9.]+)\s+source=([a-z-]+)")


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
    table = re.search(r"(?m)^\s*\[\[", text)
    at = table.start() if table else len(text)
    return text[:at] + f"{key} = {value}\n" + text[at:]


def make_config(base: str, path: Path, log: Path, mode: str,
                tokens: int, context: int | None, drafts: int,
                mtp_model: Path) -> str:
    lines = base.splitlines()
    if mode == "plain":
        lines = [line for line in lines
                 if not re.match(r"^\s*(?:mtp_[A-Za-z0-9_]+|spec_type|spec_draft_[A-Za-z0-9_]+)\s*=", line)]
    text = "\n".join(lines) + "\n"
    text = set_root_key(text, "log_path", json.dumps(str(log.resolve())))
    text = set_root_key(text, "max_new_tokens", str(tokens))
    text = set_root_key(text, "temperature", "0.0")
    text = set_root_key(text, "qwen4exp_full_graph", "true")
    text = set_root_key(text, "qwen4exp_full_graph_decode", "true")
    if context is not None:
        text = set_root_key(text, "context_size", str(context))
    if mode == "mtp":
        text = set_root_key(text, "spec_type", json.dumps("draft-mtp"))
        text = set_root_key(text, "mtp_graph", "true")
        text = set_root_key(text, "spec_draft_n_max", str(drafts))
        text = set_root_key(text, "spec_draft_model", json.dumps(str(mtp_model.resolve())))
    return text


def wait_for_idle(gpu: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while True:
        query = subprocess.run(
            ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"], capture_output=True, text=True)
        if query.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {query.stderr.strip()}")
        engine_procs = subprocess.run(["pgrep", "-x", "zeromoe"],
                                      capture_output=True, text=True).stdout.strip()
        if not query.stdout.strip() and not engine_procs:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU busy; refusing to overlap with another session: "
                               f"{query.stdout.strip() or engine_procs}")
        time.sleep(5)


def parse_run(log: Path, stdout: Path, mode: str, tokens: int) -> dict:
    text = log.read_text(errors="replace")
    wall_matches = list(WALL_RE.finditer(text))
    if not wall_matches:
        raise RuntimeError(f"{log}: missing REQUEST-DECODE-WALL marker")
    wall = WALL_RE.search(wall_matches[-1].group(0))
    n_tokens, wall_s, wall_tps = int(wall.group(1)), float(wall.group(2)), float(wall.group(3))
    if n_tokens != tokens:
        raise RuntimeError(f"{log}: decoded {n_tokens}, expected {tokens}")
    result = {"tokens": n_tokens, "decode_wall_s": wall_s, "decode_wall_tps": wall_tps}
    if mode == "mtp":
        mm = list(MTP_RE.finditer(text))
        pm = list(PROFILE_RE.finditer(text))
        if not mm or not pm:
            raise RuntimeError(f"{log}: MTP did not emit both MTP-GRAPH and MTP-PROFILE markers")
        m = MTP_RE.search(mm[-1].group(0))
        p = PROFILE_RE.search(pm[-1].group(0))
        names = ("cycles", "drafted", "accepted", "acceptance_percent", "emitted",
                 "target_steps", "batches", "batch_tokens", "replay_tokens", "draft_steps", "restores")
        vals = [int(m.group(i)) if i not in (4,) else float(m.group(i)) for i in range(1, 12)]
        result.update(dict(zip(names, vals)))
        if result["emitted"] != tokens:
            raise RuntimeError(f"{log}: MTP emitted {result['emitted']}, expected {tokens}")
        profile = PROFILE_RE.search(pm[-1].group(0))
        speculate_ms = float(profile.group(1))
        result.update({
            "steady_speculate_s": speculate_ms / 1000.0,
            "steady_speculate_tps": tokens / (speculate_ms / 1000.0),
            "window_run_ms": float(profile.group(4)),
            "draft_ms": float(profile.group(8)),
            "driver_sampling_ms": float(profile.group(9)),
        })
        builds = [tuple(map(float, b.groups()[2:])) + (int(b.group(1)), int(b.group(2)))
                  for b in BUILD_RE.finditer(text)]
        result["graph_builds"] = [
            {"weight_upload_ms": x[0], "head_upload_ms": x[1], "build_ms": x[2],
             "weights_bytes": x[3], "state_bytes": x[4]} for x in builds
        ]
        setups = list(SETUP_RE.finditer(text))
        if setups:
            s = setups[-1]
            result["graph_setup"] = {
                "factory_ms": float(s.group(1)), "runtime_ms": float(s.group(2)),
                "measure_build_ms": float(s.group(3)),
                "execution_build_ms": float(s.group(4)),
                "scheduler_alloc_ms": float(s.group(5)), "reset_ms": float(s.group(6)),
                "weights_bytes": int(s.group(7)), "state_bytes": int(s.group(8)),
                "device_workspace_bytes": int(s.group(9)), "host_workspace_bytes": int(s.group(10)),
                "post_prefill_wait_ms": float(s.group(11)), "hidden_ms": float(s.group(12)),
                "source": s.group(13),
            }
    # Keep stdout path in the JSON; the raw stdout/log files remain alongside it.
    result["stdout"] = str(stdout)
    result["engine_log"] = str(log)
    return result


def summarize(runs: list[dict], mode: str) -> dict:
    out = {"runs": len(runs),
           "median_decode_wall_tps": statistics.median(r["decode_wall_tps"] for r in runs),
           "median_decode_wall_s": statistics.median(r["decode_wall_s"] for r in runs)}
    if mode == "mtp":
        out.update({
            "median_steady_speculate_tps": statistics.median(r["steady_speculate_tps"] for r in runs),
            "median_acceptance_percent": statistics.median(r["acceptance_percent"] for r in runs),
            "median_tokens_per_cycle": statistics.median(r["emitted"] / r["cycles"] for r in runs),
        })
        if all("graph_setup" in r for r in runs):
            out["median_graph_setup_ms"] = statistics.median(
                r["graph_setup"]["factory_ms"] for r in runs)
            out["median_setup_hidden_ms"] = statistics.median(
                r["graph_setup"]["hidden_ms"] for r in runs)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, default=ROOT / "build/zeromoe")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--prompt-dir", type=Path, default=ROOT / "scratchpad/hy/tr")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mtp-model", type=Path,
                   default=Path(os.environ.get(
                       "ZEROMOE_MTP_MODEL",
                       "mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf")),
                   help="MTP draft GGUF (default: $ZEROMOE_MTP_MODEL)")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--expected-prompts", type=int, default=6)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--drafts", type=int, default=3)
    p.add_argument("--greedy-fast-path", choices=("on", "off"), default="on",
                   help="use direct argmax at temperature 0, or the probability-vector path")
    p.add_argument("--prefill-overlap", choices=("on", "off"), default="off",
                   help="request overlap when the VRAM headroom guard allows it")
    p.add_argument("--context", type=int, default=None)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--gpu", default="0")
    p.add_argument("--idle-timeout", type=int, default=0)
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()
    try:
        engine, config = args.engine.resolve(), args.config.resolve()
        prompts_dir, out_dir = args.prompt_dir.resolve(), args.out_dir.resolve()
        mtp_model = args.mtp_model.resolve()
        if not engine.is_file() or not os.access(engine, os.X_OK):
            raise ValueError(f"engine not executable: {engine}")
        if not config.is_file() or not mtp_model.is_file():
            raise ValueError("missing config or MTP sidecar")
        if not prompts_dir.is_dir():
            raise ValueError(f"prompt suite not found: {prompts_dir}")
        prompts = sorted(x for x in prompts_dir.iterdir() if x.is_file() and not x.name.startswith("."))
        if args.expected_prompts and len(prompts) != args.expected_prompts:
            raise ValueError(f"expected {args.expected_prompts} prompts, found {len(prompts)}")
        if args.tokens <= 0 or args.repeats <= 0 or args.drafts < 1:
            raise ValueError("tokens, repeats, and drafts must be positive")
        prompt_text = {}
        for prompt in prompts:
            text = prompt.read_text().strip()
            if not text or "\n" in text:
                raise ValueError(f"prompt must be a nonempty single line: {prompt}")
            prompt_text[prompt.name] = text
        if out_dir.exists():
            raise ValueError(f"output directory already exists: {out_dir}")
        subprocess.run([sys.executable, str(ROOT / "tools/build_provenance.py"),
                        "check", "--binary", str(engine), "--repo", str(ROOT)],
                       cwd=ROOT, check=True)
        out_dir.mkdir(parents=True)
        (out_dir / "build-info.txt").write_text(Path(str(engine) + ".build-info.txt").read_text())
        manifest = {
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "engine": str(engine), "engine_sha256": sha256(engine),
            "config": str(config), "config_sha256": sha256(config),
            "mtp_sidecar": str(mtp_model), "mtp_sha256": sha256(mtp_model),
            "tokens": args.tokens, "repeats": args.repeats, "drafts": args.drafts,
            "context_override": args.context, "threads": args.threads, "gpu": args.gpu,
            "prompt_hashes": {x.name: hashlib.sha256(prompt_text[x.name].encode()).hexdigest() for x in prompts},
            "bench_ignore_stop": True,
        "greedy_fast_path": args.greedy_fast_path,
        "prefill_overlap": args.prefill_overlap,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        base = config.read_text()
        base_env = {k: v for k, v in os.environ.items() if not k.startswith("ZEROMOE_")}
        base_env.update({
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "OMP_NUM_THREADS": str(args.threads),
            "ZEROMOE_QWEN4EXP_FULL_GRAPH": "1",
            "ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE": "1",
            "ZEROMOE_QWEN_CPU_THREADS": str(args.threads),
            "ZEROMOE_BENCH_IGNORE_STOP": "1",
            "ZEROMOE_BENCH_SILENT": "1",
        "ZEROMOE_MTP_GREEDY_FAST_PATH": "1" if args.greedy_fast_path == "on" else "0",
        "ZEROMOE_MTP_PREFILL_OVERLAP": "1" if args.prefill_overlap == "on" else "0",
        })
        results = {}
        for prompt_file in prompts:
            prompt_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", prompt_file.stem)
            prompt_out = out_dir / prompt_name
            prompt_out.mkdir()
            prompt = prompt_text[prompt_file.name]
            cells = {"plain": [], "mtp": []}
            for rep in range(args.repeats):
                modes = ["plain", "mtp"] if rep % 2 == 0 else ["mtp", "plain"]
                for mode in modes:
                    run_dir = prompt_out / f"rep-{rep+1:02d}" / mode
                    run_dir.mkdir(parents=True)
                    env = base_env.copy()
                    log = run_dir / "engine.log"
                    stdout = run_dir / "stdout.txt"
                    token_file = run_dir / "generated.tokens"
                    env["ZEROMOE_GENERATED_TOKEN_FILE"] = str(token_file)
                    text = make_config(base, run_dir / "config.toml", log,
                                       mode, args.tokens, args.context, args.drafts, mtp_model)
                    (run_dir / "config.toml").write_text(text)
                    wait_for_idle(args.gpu, args.idle_timeout)
                    print(f"{prompt_file.name} rep={rep+1} mode={mode}", flush=True)
                    with stdout.open("w") as out:
                        subprocess.run([str(engine), "--graph-device-expert-layers", "0", "--graph-device-expert-trailing", "1", "--config", str(run_dir / "config.toml")],
                                       input=prompt + "\n:q\n", text=True,
                                       stdout=out, stderr=subprocess.STDOUT,
                                       cwd=ROOT, env=env, timeout=args.timeout, check=True)
                    raw = [int(x) for x in token_file.read_text().split()] if token_file.exists() else []
                    if mode == "plain" and len(raw) != args.tokens:
                        raise RuntimeError(f"{prompt_file.name}/plain: generated {len(raw)}, expected {args.tokens}")
                    result = parse_run(log, stdout, mode, args.tokens)
                    result["token_dump_count"] = len(raw)
                    result["generated_token_count"] = (
                        len(raw) if mode == "plain" else result["emitted"])
                    if mode == "mtp" and len(raw) not in (0, args.tokens):
                        raise RuntimeError(f"{prompt_file.name}/mtp: token dump has {len(raw)} entries")
                    result["generated_tokens_sha256"] = sha256(token_file) if token_file.exists() else None
                    cells[mode].append(result)
            prompt_summary = {mode: summarize(runs, mode) for mode, runs in cells.items()}
            (prompt_out / "summary.json").write_text(json.dumps({"runs": cells, "summary": prompt_summary}, indent=2) + "\n")
            results[prompt_file.name] = prompt_summary
            print(f"{prompt_file.name}: plain={prompt_summary['plain']['median_decode_wall_tps']:.2f} "
                  f"mtp={prompt_summary['mtp']['median_decode_wall_tps']:.2f} tok/s", flush=True)
        (out_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"results: {out_dir}")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
