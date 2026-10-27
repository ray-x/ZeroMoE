#!/usr/bin/env python3
"""Targeted phase-transition correctness/memory test, not the frozen performance gate.

Run on an idle CUDA host: python3 bench/phase_cache_transition.py --model mimo|ling --model-path <model.gguf>
Runs a short, chunked long, then repeated short request in each configuration.
Captures logits before submitting the next request, so prompt resets and state
survival across graph rebuilds are checked without autoregressive divergence.
"""
import argparse
import array
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def all_logits(path):
    frames = []
    with path.open("rb") as f:
        while header := f.read(8):
            step, n = struct.unpack("<II", header)
            values = array.array("f")
            values.fromfile(f, n)
            if step != len(frames) or not all(math.isfinite(v) for v in values):
                raise AssertionError(f"invalid logits: {path} step={step}")
            frames.append(values)
    if not frames:
        raise AssertionError(f"missing logits: {path}")
    return frames


def first_logits(path):
    return all_logits(path)[0]


def compare(a, b):
    if len(a) != len(b):
        raise AssertionError("logit vocabulary changed")
    dot = math.fsum(x*y for x, y in zip(a, b))
    norm = math.sqrt(math.fsum(x*x for x in a) * math.fsum(y*y for y in b))
    return {"cosine": dot / norm, "max_abs": max(abs(x-y) for x, y in zip(a, b))}


def compare_requests(a, b):
    a, b = all_logits(a), all_logits(b)
    if len(a) != len(b):
        raise AssertionError(f"teacher-forced step count differs: {len(a)} != {len(b)}")
    steps = [compare(x, y) for x, y in zip(a, b)]
    return {"steps": len(steps), "cosine": min(x["cosine"] for x in steps),
            "max_abs": max(x["max_abs"] for x in steps)}


def prepare(model, model_path, directory, isolate_assist=False, gpu0_cache_gb=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ZEROMOE_")}
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    graph_cli = []
    if model == "mimo":
        text = (ROOT / "configs/bench/mimo26_flash_gsq_rco_3bit_ingraph_dual.toml").read_text()
        profile = ROOT / "bench/profiles/mimo26_combined.usage"
        env.update(ZEROMOE_DISABLE_ROUTER_BF16="1")
        graph_cli = ["--graph-offload-ddr-gib", "100",
                     "--graph-offload-usage-file", str(profile),
                     "--graph-prefill-chunk", "512"]
        cap = 110
    else:
        text = (ROOT / "configs/bench/ling/reference-audit/config.toml").read_text()
        profile = ROOT / "bench/ling/reference-audit/usage-hi-there-robust4.pin"
        text = re.sub(r"^pinned_vram_capacity = .*", "pinned_vram_capacity = 3700", text, flags=re.M)
        env.update(ZEROMOE_GLM5NEXT_FULL_GRAPH_PREFILL="1",
                   ZEROMOE_FULL_GRAPH_USAGE_FILE=str(profile),
                   ZEROMOE_GLM5NEXT_CPU_THREADS="10",
                   ZEROMOE_GLM5NEXT_BATCHED_NEGATIVE_SKIP="1")
        graph_cli = [
            "--graph-dynamic-cache", "0",
            "--graph-gpu-assist", "3",
            "--graph-gpu-assist2-slots", "1",
            "--graph-early-ready", "1",
            "--graph-offload-first-layer", "4",
            "--graph-offload-model", str(model_path.resolve()),
            "--graph-offload-usage-file", str(profile),
            "--graph-offload-ddr-gib", "64",
            "--graph-offload-pinned-fraction", "0.5",
            "--graph-offload-byte-cap", "true",
            "--graph-offload-pin-dma", "true",
            "--graph-offload-async", "true",
            "--graph-prefill-chunk", "256",
            "--graph-prefill-op-offload", "false",
        ]
        cap = 95
        if isolate_assist:
            graph_cli += ["--graph-gpu-assist", "0",
                          "--graph-gpu-assist2-slots", "0"]
            text = "# phase-validation: adaptive GPU assist disabled in both reference and candidate\n" + text
    text, n_model = re.subn(
        r"^model_path = .*$",
        f"model_path = {json.dumps(str(model_path.resolve()))}",
        text, count=1, flags=re.M,
    )
    if n_model != 1:
        raise AssertionError("cannot find model_path in the base config")
    if gpu0_cache_gb is not None:
        # Equivalent-placement control: the same experts the extension would hold live in the permanent GPU 0 cache.
        text, n = re.subn(r"^moe_cache_gb = [0-9.]+", f"moe_cache_gb = {gpu0_cache_gb}", text, count=1, flags=re.M)
        if n != 1:
            raise AssertionError("cannot find GPU 0 moe_cache_gb to enlarge")
    text = re.sub(r"^context_size = .*", "context_size = 2048", text, flags=re.M)
    text = re.sub(r"^pin_file = .*", f'pin_file = "{profile}"', text, flags=re.M)
    config = directory / "test.toml"
    config.write_text(text)
    return env, config, cap


def run_case(args, name, cache_mib, prompts, teacher_source=None, gpu0_cache_gb=None):
    directory = args.output / name
    directory.mkdir(parents=True, exist_ok=True)
    env, config, cap = prepare(args.model, args.model_path, directory, args.isolate_assist, gpu0_cache_gb)
    env.update(ZEROMOE_DECODE_LOGITS_DUMP=str(directory / "current.bin"),
               ZEROMOE_GENERATED_TOKEN_FILE=str(directory / "current.tokens"))
    if teacher_source:
        teacher = directory / "teacher"
        teacher.mkdir()
        for index, reference in enumerate((0, 1, 0)):
            shutil.copyfile(teacher_source / f"request-{reference}.tokens", teacher / f"request-{index}.tokens")
        env["ZEROMOE_TEACHER_FORCE_REQUEST_DIR"] = str(teacher)
    command = [str(ROOT / "scripts/run_capped.sh"), "--limit-gb", str(cap), "--", str(args.binary),
               "--config", str(config), "--predict", str(args.ngen), "--temp", "0", "--log-file", str(directory / "engine.log"),
               "--graph-phase-cache-mib", str(cache_mib), "--graph-phase-cache-reserve-mib", "512",
               *graph_cli]
    samples = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
            if p.returncode == 0:
                samples.append([time.monotonic(), *[int(x) for x in p.stdout.splitlines()]])
            stop.wait(0.5)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    with (directory / "console.log").open("w") as console:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                   stdout=console, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            for index, prompt in enumerate(prompts):
                process.stdin.write(prompt + "\n"); process.stdin.flush()
                deadline = time.monotonic() + args.timeout
                while time.monotonic() < deadline:
                    log = (directory / "engine.log").read_text(errors="replace") if (directory / "engine.log").exists() else ""
                    if log.count("[REQUEST-DECODE-WALL]") > index:
                        break
                    if process.poll() is not None:
                        raise RuntimeError(f"{name} exited {process.returncode}; see {directory}")
                    time.sleep(0.2)
                else:
                    raise TimeoutError(f"{name} request {index} timed out; see {directory}")
                shutil.copyfile(directory / "current.bin", directory / f"request-{index}.bin")
                shutil.copyfile(directory / "current.tokens", directory / f"request-{index}.tokens")
                print(f"{args.model} {name} request {index}: completed", flush=True)
            process.stdin.close()
            code = process.wait(timeout=60)
            if code:
                raise RuntimeError(f"{name} shutdown exited {code}; see {directory}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            stop.set(); worker.join(timeout=10)
            (directory / "vram-samples.json").write_text(json.dumps(samples))
    log = (directory / "engine.log").read_text(errors="replace")
    events = re.findall(r"\[PHASE-CACHE\] decode generation=(\d+) device=(\d+) experts=(\d+) bytes=(\d+) refill_ms=([\d.]+) SSD_loads=0", log)
    releases = re.findall(r"\[PHASE-CACHE\] prefill generation=(\d+) released_bytes=(\d+) release_ms=([\d.]+)", log)
    if cache_mib:
        if len(events) < 3 or len(releases) < 2:
            raise AssertionError(f"missing phase transitions: {events}, {releases}")
        if len({(e[2], e[3]) for e in events}) != 1:
            raise AssertionError("decode extension identities/capacity changed across requests")
        if not all(int(events[i+1][0]) > int(events[i][0]) for i in range(len(events)-1)):
            raise AssertionError("placement generation did not advance")
    loads = re.findall(r"\[(?:MIMO2-OFFLOAD|GGML-GLM5NEXT-OFFLOAD)\] shutdown:.*?loads=(\d+)", log)
    if not loads:
        loads = re.findall(r"\[(?:MIMO2|GLM5NEXT)-LANE\].*?loads=(\d+)", log)
    if not loads or (args.model == "mimo" and any(int(n) for n in loads)):
        raise AssertionError(f"missing residency evidence or unexpected decode expert loads: {loads}")
    reset = compare_requests(directory / "request-0.bin", directory / "request-2.bin")
    if reset["cosine"] < 0.99999:
        raise AssertionError(f"repeated prompt changed after graph/state transitions: {reset}")
    return {"events": events, "releases": releases, "reset": reset,
            "decode_store_loads": max(map(int, loads)),
            "peak_vram_mib": [max(s[i+1] for s in samples) for i in range(2)] if samples else []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mimo", "ling"), required=True)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="model file or shard directory for the selected model")
    parser.add_argument("--binary", type=Path, default=ROOT / "build_mimo/zeromoe")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-mib", type=int, default=1024)
    parser.add_argument("--ngen", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--control-output", type=Path,
                        help="Reuse a completed control case directory after a phase-only fix; verifies its configuration")
    parser.add_argument("--control-only", action="store_true", help="Collect reference requests without enabling the extension")
    parser.add_argument("--isolate-assist", action="store_true",
                        help="Disable adaptive GPU assist in Ling control and candidate to isolate cache placement")
    parser.add_argument("--equivalent-gpu0-gb", type=float,
                        help="Also run a fixed-cache control whose GPU 0 cache is this many GB (base + extension bytes): "
                             "the placement-noise baseline, and the closest equivalent-placement reference for the candidate")
    parser.add_argument("--equivalent-only", action="store_true",
                        help="With --control-output: run only the equivalent-placement control and compare it to the control")
    args = parser.parse_args()
    args.binary = args.binary.resolve(); args.output = args.output.resolve()
    args.model_path = args.model_path.resolve()
    if not args.model_path.exists():
        parser.error(f"model path does not exist: {args.model_path}")
    args.output.mkdir(parents=True, exist_ok=False)
    short = "Explain why the sky appears blue in one sentence."
    long = ("The red parcel belongs to Ada. The blue parcel belongs to Ben. " * 48 +
            "Who owns the red parcel? Answer with the name only.")
    prompts = [short, long, short]
    result = {"model": args.model, "binary": str(args.binary)}
    if args.control_output:
        control = args.control_output.resolve()
        check = args.output / "control-config-check"
        check.mkdir()
        _, config, _ = prepare(args.model, args.model_path, check, args.isolate_assist)
        if config.read_text() != (control / "test.toml").read_text():
            raise AssertionError("reused control has a different model/workload configuration")
        for index in range(2):
            first_logits(control / f"request-{index}.bin")
        result["control"] = {"reused_from": str(control), "oracle_requests": [0, 1, 0]}
    else:
        result["control"] = run_case(args, "control", 0, prompts)
        control = args.output / "control"
    if args.control_only:
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.equivalent_gpu0_gb:
        result["equivalent"] = run_case(args, "equivalent", 0, prompts, control, args.equivalent_gpu0_gb)
        result["equivalent_vs_control"] = [
            compare_requests(control / f"request-{(0, 1, 0)[i]}.bin", args.output / "equivalent" / f"request-{i}.bin")
            for i in range(3)]
        print("placement-noise baseline (equivalent vs control):", json.dumps(result["equivalent_vs_control"]), flush=True)
        if args.equivalent_only:
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            return
    result["phase"] = run_case(args, "phase", args.cache_mib, prompts, control)
    if args.equivalent_gpu0_gb:
        result["phase_vs_equivalent"] = [
            compare_requests(args.output / "equivalent" / f"request-{i}.bin", args.output / "phase" / f"request-{i}.bin")
            for i in range(3)]
        print("phase vs equivalent placement:", json.dumps(result["phase_vs_equivalent"]), flush=True)
    result["parity"] = []
    for index in range(3):
        reference = (0, 1, 0)[index]
        comparison = compare_requests(control / f"request-{reference}.bin",
                                      args.output / "phase" / f"request-{index}.bin")
        # MiMo must stay within 0.999 of the control. Ling's IQ1_S/IQ2_XXS experts amplify the CPU-vs-CUDA activation
        # rounding difference (about 0.5-0.9% per op, exact kernels), so moving experts between CPU and GPU cannot
        # hold 0.999 there; Ling is gated by scripts/check_kernel_numerics.sh and by bitwise-identical repeats below.
        if args.model == "mimo" and comparison["cosine"] < 0.999:
            raise AssertionError(f"phase/control logit mismatch request {index}: {comparison}")
        result["parity"].append(comparison)
    if args.model == "ling":
        repeat = compare_requests(args.output / "phase" / "request-0.bin", args.output / "phase" / "request-2.bin")
        result["repeat"] = repeat
        if repeat["cosine"] < 1.0 or repeat["max_abs"] != 0.0:
            raise AssertionError(f"repeated short request is not bitwise identical after the transitions: {repeat}")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
