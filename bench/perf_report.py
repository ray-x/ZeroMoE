#!/usr/bin/env python3
"""Collect one standard performance record per model/config on this machine, and render records as Markdown.

  bench/perf_report.py run --name qwen38-flash --config configs/qwen38-flash.toml \\
      --model /path/to/model --binary build/zeromoe \\
      --env ZEROMOE_QWEN4EXP_FULL_GRAPH=1 --set omp_max_threads=10 --note ram="DDR4-2666 4ch"
  bench/perf_report.py render bench/results/*.json

What a record holds (so models can be compared and summarised into README / PERFORMANCE.md):
  * setup: CPU, RAM, GPUs (PCIe link, power limit), OS, CUDA driver/toolkit, ZeroMoE commit and whether the tree was
    dirty, ggml commit and whether the ggml sources were modified, model files, the config used (model path redacted),
    every setting/environment variable applied, and the engine's own resolved-settings log lines;
  * prefill speed per prompt length (tokens/s) and their average;
  * decode speed per prompt length: average, peak (best 32-step window), and the median over repeats;
  * long decode: one long generation split into equal buckets to show whether the speed is stable;
  * second-request decode setup time (the cost a chat turn pays after its prefill).

Method: each measurement waits for a quiet machine, runs the engine in the foreground, and is accepted only if it
generated exactly the requested number of tokens and no other process disturbed the decode window (outside CPU during
decode, another zeromoe, or a compiler). Rejected runs are retried and kept in <name>.attempts.jsonl.
Decode speed is 1000 / mean(iter_total ms) over steps >= 3, the same definition as bench/qwen_decode_baselines.sh.
"""
import argparse, glob, hashlib, json, os, platform, re, statistics as st, subprocess, sys, time

HZ = os.sysconf("SC_CLK_TCK")
NOISY = {"cicc", "cc1plus", "cc1", "nvcc", "ptxas", "fatbinary", "ld", "collect2", "ninja", "make", "cmake", "rsync"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = {"short": "bench/prompts/short.txt", "mid": "bench/prompts/mid.txt", "long": "bench/prompts/long.txt"}
STD_NGEN = 200
BUCKETS = 6


# ---------------------------------------------------------------- machine and build description
def sh(cmd, default=""):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip() or default
    except Exception:
        return default


def system_info(visible=None):
    lscpu = {k.strip(): v.strip() for k, _, v in (l.partition(":") for l in sh("lscpu").splitlines())}
    flags = lscpu.get("Flags", "").split()
    mem_kb = int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read())[1])
    gpus = []
    q = "name,memory.total,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,power.limit,clocks.max.sm"
    for line in sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader,nounits").splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 8:
            gpus.append(dict(zip(["name", "memory_mib", "driver", "pcie_gen", "pcie_gen_max", "pcie_width", "power_limit_w",
                                  "max_sm_mhz"], f)))
    vis = visible if visible is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    used = {int(x) for x in vis.split(",") if x.strip().isdigit()} if vis else set(range(len(gpus)))
    for i, g in enumerate(gpus):
        g["used"] = i in used
    return {
        "cpu": lscpu.get("Model name", platform.processor()),
        "cpu_threads": int(lscpu.get("CPU(s)", os.cpu_count() or 0)),
        "cpu_cores": int(lscpu.get("Core(s) per socket", 0) or 0) * int(lscpu.get("Socket(s)", 1) or 1),
        "cpu_avx512": {k: (k in flags) for k in ("avx512f", "avx512_vnni", "avx512_vbmi", "avx2")},
        "ram_total_gib": round(mem_kb / 1048576, 1),
        "gpus": gpus,
        "os": sh("grep PRETTY_NAME /etc/os-release | cut -d'\"' -f2", platform.platform()),
        "kernel": platform.release(),
        "cuda_toolkit": sh("nvcc --version | grep release | sed 's/.*release //; s/,.*//'", "unknown"),
    }


def build_info(binary, binary_commit=None):
    def git(args):
        return sh(f"git -C {ROOT} {args}")
    ggml = git("ls-tree HEAD third_party/llama.cpp").split()
    sub = os.path.join(ROOT, "third_party/llama.cpp")
    head_time = int(git("log -1 --format=%ct HEAD") or 0)
    newer = os.path.getmtime(binary) >= head_time
    built_from = binary_commit or (git("rev-parse --short=12 HEAD") if newer else "unknown (binary is older than HEAD; pass --binary-commit)")
    return {
        "zeromoe_commit": built_from,
        "repo_head": git("rev-parse --short=12 HEAD"),
        "zeromoe_tree_dirty": (bool(git("status --porcelain --untracked-files=no"))
                               if built_from == git("rev-parse --short=12 HEAD") else None),
        "ggml_commit": ggml[2][:12] if len(ggml) >= 3 else "unknown",
        "ggml_sources_modified": (bool(sh(f"git -C {sub} status --porcelain --untracked-files=no"))
                                  if os.path.exists(os.path.join(sub, ".git")) else "not checked out"),
        "binary": os.path.abspath(binary),
        "binary_mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(binary))),
    }


def model_info(path):
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "*.gguf")))
    return [{"file": os.path.basename(f), "gib": round(os.path.getsize(f) / 1073741824, 2)} for f in files]


def config_info(cfg_path, text):
    top = {}
    for line in text.splitlines():
        if line.startswith("[["):
            break
        m = re.match(r"^([A-Za-z0-9_]+)\s*=\s*(.+?)\s*(#.*)?$", line)
        if m and m[1] != "model_path":
            top[m[1]] = m[2].strip('"')
    return {"file": os.path.relpath(cfg_path, ROOT), "sha256": hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()[:12],
            "settings": top}


# ---------------------------------------------------------------- quiet-machine handling
def busy_ticks():
    v = list(map(int, open("/proc/stat").readline().split()[1:]))
    return v[0] + v[1] + v[2] + v[5] + v[6] + v[7]


def proc_ticks(pid):
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12])
    except OSError:
        return 0


def scan(own_pid):
    other, noisy = 0, set()
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == own_pid:
            continue
        try:
            comm = open(f"/proc/{d}/comm").read().strip()
        except OSError:
            continue
        if comm == "zeromoe":
            other += 1
        elif comm in NOISY:
            noisy.add(comm)
    return other, noisy


def wait_quiet(seconds=10, limit=1500):
    t0, run, pb = time.time(), 0, busy_ticks()
    while time.time() - t0 < limit:
        time.sleep(1)
        b = busy_ticks()
        ext, pb = (b - pb) / HZ, b
        other, noisy = scan(-1)
        run = run + 1 if (ext < 0.8 and other == 0 and not noisy) else 0
        if run >= seconds:
            return True
    return False


def log_time(log, pat, t0, last=False):
    ms = re.findall(r"\[(\d+):(\d+):(\d+)\.(\d+)\] " + pat, log)
    if not ms:
        return None
    m = ms[-1] if last else ms[0]
    d = time.localtime(t0)
    return time.mktime((d.tm_year, d.tm_mon, d.tm_mday, int(m[0]), int(m[1]), int(m[2]), 0, 0, -1)) + int(m[3]) / 1000.0


# ---------------------------------------------------------------- one engine run
def set_config_key(cfg, key, value):
    """Set a top-level TOML key; a new key goes before the first [[table]] (keys after it belong to that table)."""
    pat = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
    if pat.search(cfg):
        return pat.sub(f"{key} = {value}", cfg, count=1)
    i = cfg.find("\n[[")
    return cfg + f"\n{key} = {value}\n" if i < 0 else cfg[:i] + f"\n{key} = {value}\n" + cfg[i:]


def run_engine(args, tag, prompt_file, ngen, cfg_text, env_extra):
    os.makedirs(args.out, exist_ok=True)
    logdir = os.path.join(args.out, f"{args.name}.logs")
    os.makedirs(logdir, exist_ok=True)
    cfg = re.sub(r"(?m)^model_path\s*=.*$", f'model_path = "{args.model}"', cfg_text)
    cfg = re.sub(r"(?m)^log_path\s*=.*$", f'log_path = "{logdir}/{tag}.engine.log"', cfg)
    for kv in args.set:
        k, _, v = kv.partition("=")
        cfg = set_config_key(cfg, k, v)
    cfg_file = f"{logdir}/{tag}.toml"
    open(cfg_file, "w").write(cfg)
    env = dict(os.environ, ZEROMOE_TOKEN_LOOP_DIAG="1", ZEROMOE_BENCH_IGNORE_STOP="1", **env_extra)
    text = open(os.path.join(ROOT, prompt_file)).read().replace("\n", "") + "\n"
    out = open(f"{logdir}/{tag}.out", "w")
    t0 = time.time()
    p = subprocess.Popen([args.binary, "--config", cfg_file, "--ngen", str(ngen)], cwd=ROOT, env=env, stdin=subprocess.PIPE,
                         stdout=out, stderr=subprocess.STDOUT, text=True)
    p.stdin.write(text)
    p.stdin.close()
    samples, events = [], []
    pb, pp = busy_ticks(), proc_ticks(p.pid)
    while p.poll() is None:
        time.sleep(1)
        b, q = busy_ticks(), proc_ticks(p.pid)
        now = time.time()
        samples.append((now, max(0.0, ((b - pb) - (q - pp)) / HZ)))
        pb, pp = b, q
        o, nz = scan(p.pid)
        if o or nz:
            events.append((now, o, sorted(nz)))
    out.close()
    log = open(f"{logdir}/{tag}.engine.log", errors="replace").read()
    return parse(log, ngen, t0, samples, events)


def window_rate(ms):
    return 1000.0 / (sum(ms) / len(ms))


def parse(log, ngen, t0, samples, events):
    steps = [float(m.group(2)) for m in re.finditer(r"step=(\d+) iter_total=([\d.]+)ms", log) if int(m.group(1)) >= 3]
    pre = re.findall(r"\[PREFILL-DONE\] (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)", log)
    tok = re.findall(r"\[REQUEST\] (\d+) tokens", log)
    setup = [float(x) for x in re.findall(r"\[[A-Z0-9-]*DECODE-SETUP\] prepare_total=([\d.]+)ms", log)]
    wall = re.findall(r"REQUEST-DECODE-WALL\] tokens=\d+ wall=([\d.]+)s rate=([\d.]+)", log)
    a = log_time(log, r"\[PREFILL-DONE\] \d+ tokens", t0)
    b = log_time(log, r"\[REQUEST\] \d+ tokens", t0, last=True)
    ws = [e for (t, e) in samples if a and b and t > a and t - 1.0 < b]
    ev = [x for x in events if a and b and a - 1.0 <= x[0] <= b + 1.0]
    r = {
        "tokens": int(tok[-1]) if tok else 0,
        "prompt_tokens": int(pre[-1][0]) if pre else None,
        "prefill_s": float(pre[-1][1]) if pre else None,
        "prefill_tok_s": float(pre[-1][2]) if pre else None,
        "decode_avg_tok_s": round(window_rate(steps), 2) if steps else None,
        "decode_peak_tok_s": round(max(window_rate(steps[i:i + 32]) for i in range(max(1, len(steps) - 31))), 2) if len(steps) >= 32 else None,
        "decode_steps": len(steps),
        "setup_ms": setup,
        "request_rate_tok_s": float(wall[-1][1]) if wall else None,
        "outside_cpu_mean": round(sum(ws) / len(ws), 2) if ws else None,
        "outside_cpu_max": round(max(ws), 2) if ws else None,
        "other_zeromoe": sum(x[1] for x in ev),
        "noisy_procs": sorted({n for x in ev for n in x[2]}),
    }
    n = len(steps) // BUCKETS
    r["bucket_tok_s"] = [round(window_rate(steps[i * n:(i + 1) * n]), 2) for i in range(BUCKETS)] if n >= 20 else None
    r["complete"] = r["tokens"] == ngen and r["decode_avg_tok_s"] is not None
    r["quiet"] = bool(ws) and r["outside_cpu_mean"] <= 0.5 and r["outside_cpu_max"] <= 1.0 and not r["other_zeromoe"] and not r["noisy_procs"]
    return r


# ---------------------------------------------------------------- driver
def plan(args):
    p = [("std", name, STD_NGEN, rep) for name in PROMPTS for rep in range(1, args.reps + 1)]
    if args.long_ngen:
        p += [("long_decode", "short", args.long_ngen, rep) for rep in range(1, args.long_reps + 1)]
    return p


def cmd_run(args):
    cfg_path = os.path.join(ROOT, args.config) if not os.path.isabs(args.config) else args.config
    cfg_text = open(cfg_path).read()
    env_extra = dict(kv.split("=", 1) for kv in args.env)
    os.makedirs(args.out, exist_ok=True)
    attempts_f = os.path.join(args.out, f"{args.name}.attempts.jsonl")
    accepted_f = os.path.join(args.out, f"{args.name}.runs.jsonl")
    done = {(r["kind"], r["prompt"], r["rep"]) for r in map(json.loads, open(accepted_f))} if os.path.exists(accepted_f) else set()
    t_end = time.time() + args.deadline
    for kind, prompt, ngen, rep in plan(args):
        if (kind, prompt, rep) in done:
            continue
        for attempt in range(1, 4):
            if time.time() > t_end - 100:
                print(f"time budget reached before {kind}/{prompt}/r{rep}; rerun the same command to continue")
                return
            if not wait_quiet():
                print("machine never became quiet; stopping")
                return
            tag = f"{kind}_{prompt}_r{rep}_a{attempt}"
            r = run_engine(args, tag, PROMPTS[prompt], ngen, cfg_text, env_extra)
            r.update(kind=kind, prompt=prompt, rep=rep, attempt=attempt, ngen=ngen)
            ok = r["complete"] and r["quiet"]
            r["accepted"] = ok
            open(attempts_f, "a").write(json.dumps(r) + "\n")
            print(f"{tag}: {'OK' if ok else ('INCOMPLETE' if not r['complete'] else 'DISTURBED')} tokens={r['tokens']}/{ngen} "
                  f"prefill={r['prefill_tok_s']} decode={r['decode_avg_tok_s']} peak={r['decode_peak_tok_s']} "
                  f"setup={r['setup_ms']} outside={r['outside_cpu_mean']}/{r['outside_cpu_max']}", flush=True)
            if ok:
                open(accepted_f, "a").write(json.dumps(r) + "\n")
                break
        else:
            print(f"{kind}/{prompt}/r{rep}: no clean run after 3 attempts; stopping")
            return
    finalize(args, cfg_path, cfg_text, env_extra, accepted_f, attempts_f)


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 2) if xs else None


def finalize(args, cfg_path, cfg_text, env_extra, accepted_f, attempts_f):
    runs = [json.loads(l) for l in open(accepted_f)]
    attempts = [json.loads(l) for l in open(attempts_f)]
    first_log = open(os.path.join(args.out, f"{args.name}.logs", "std_short_r1_a1.engine.log"), errors="replace").read() \
        if os.path.exists(os.path.join(args.out, f"{args.name}.logs", "std_short_r1_a1.engine.log")) else ""
    keep = ("[SETTINGS]", "[GRAPH-CAPS]", "[GRAPH-TUNING]", "device expert arena", "soft expert tier", "[GRAPH-RAM] expert-arena")
    engine_lines = [re.sub(r"^\[[\d:.]+\] ", "", l)[:300] for l in first_log.splitlines() if any(k in l for k in keep)][:24]
    std = [r for r in runs if r["kind"] == "std"]
    by_prompt = {}
    for name in PROMPTS:
        rs = [r for r in std if r["prompt"] == name]
        if rs:
            by_prompt[name] = {
                "prompt_tokens": rs[0]["prompt_tokens"], "n": len(rs),
                "prefill_tok_s": med(r["prefill_tok_s"] for r in rs),
                "prefill_min": min(r["prefill_tok_s"] for r in rs), "prefill_max": max(r["prefill_tok_s"] for r in rs),
                "decode_avg_tok_s": med(r["decode_avg_tok_s"] for r in rs),
                "decode_min": min(r["decode_avg_tok_s"] for r in rs), "decode_max": max(r["decode_avg_tok_s"] for r in rs),
                "decode_peak_tok_s": med(r["decode_peak_tok_s"] for r in rs),
                "second_setup_s": round(med(r["setup_ms"][-1] for r in rs if len(r["setup_ms"]) > 1) / 1000, 2)
                if any(len(r["setup_ms"]) > 1 for r in rs) else None,
                "request_rate_tok_s": med(r["request_rate_tok_s"] for r in rs),
            }
    long_runs = [r for r in runs if r["kind"] == "long_decode"]
    long = None
    if long_runs:
        bk = [r["bucket_tok_s"] for r in long_runs if r["bucket_tok_s"]]
        per_bucket = [med(b[i] for b in bk) for i in range(BUCKETS)] if bk else None
        long = {"tokens": long_runs[0]["ngen"], "n": len(long_runs), "avg_tok_s": med(r["decode_avg_tok_s"] for r in long_runs),
                "bucket_tokens": long_runs[0]["ngen"] // BUCKETS, "bucket_tok_s": per_bucket,
                "drift_pct": round((per_bucket[-1] / per_bucket[0] - 1) * 100, 1) if per_bucket else None}
    pm = [v["prefill_tok_s"] for v in by_prompt.values()]
    record = {
        "schema": 1, "name": args.name, "date": time.strftime("%Y-%m-%d"),
        "system": system_info(env_extra.get("CUDA_VISIBLE_DEVICES")), "build": build_info(args.binary, args.binary_commit), "model": model_info(args.model),
        "config": config_info(cfg_path, cfg_text),
        "overrides": args.set, "environment": {**{k: v for k, v in os.environ.items() if re.match(r"(ZEROMOE_|CUDA_VISIBLE|OMP_|GGML_)", k)},
                                              **env_extra, "ZEROMOE_TOKEN_LOOP_DIAG": "1", "ZEROMOE_BENCH_IGNORE_STOP": "1"},
        "notes": dict(kv.split("=", 1) for kv in args.note), "engine_log_lines": engine_lines,
        "protocol": {"standard_ngen": STD_NGEN, "reps": args.reps, "prompts": PROMPTS, "decode_definition":
                     "1000 / mean(iter_total ms), steps >= 3", "peak_definition": "best 32-step window",
                     "attempts": len(attempts), "rejected": sum(1 for a in attempts if not a["accepted"])},
        "summary": {"prefill_avg_tok_s": round(st.mean(pm), 1) if pm else None, "by_prompt": by_prompt, "long_decode": long},
        "runs": runs,
    }
    path = os.path.join(args.out, f"{args.name}.json")
    json.dump(record, open(path, "w"), indent=1)
    print("wrote", path)


# ---------------------------------------------------------------- render
def cmd_render(args):
    recs = [json.load(open(f)) for f in args.files]
    print("## Setup\n")
    for r in recs:
        s, b, c = r["system"], r["build"], r["config"]
        gpus = "; ".join((f"{g['name']} {int(float(g['memory_mib'])/1024)} GiB, PCIe gen {g['pcie_gen_max']} x{g['pcie_width']} "
                          f"(link capability; the idle link drops to a lower gen), {float(g['power_limit_w']):.0f} W")
                         if g.get("used", True) else f"{g['name']} (present, not used in this run)" for g in s["gpus"])
        v = s["cpu_avx512"]
        print(f"**{r['name']}** ({r['date']})\n")
        print(f"- CPU: {s['cpu']}, {s['cpu_cores']} cores / {s['cpu_threads']} threads"
              f" (AVX-512 {'yes' if v['avx512f'] else 'no'}, VNNI {'yes' if v['avx512_vnni'] else 'no'}, VBMI {'yes' if v['avx512_vbmi'] else 'no'})")
        print(f"- RAM: {s['ram_total_gib']} GiB" + "".join(f", {k} {v}" for k, v in r["notes"].items()))
        print(f"- GPU: {gpus}")
        print(f"- OS: {s['os']}, kernel {s['kernel']}, CUDA toolkit {s['cuda_toolkit']}")
        print(f"- ZeroMoE `{b['zeromoe_commit']}`{' (built from a tree with uncommitted changes)' if b['zeromoe_tree_dirty'] else ''}, "
              f"ggml `{b['ggml_commit']}`{' (ggml sources modified)' if b['ggml_sources_modified'] is True else ''}, binary built {b['binary_mtime']}")
        print(f"- Model: " + ", ".join(f"{m['file']} ({m['gib']} GiB)" for m in r["model"]))
        keys = ("context_size", "omp_max_threads", "host_ram_gb", "ram_pinned_budget_gb", "qwen4exp_option_b_global_top_k")
        shown = ", ".join(f"{k}={c['settings'][k]}" for k in keys if k in c["settings"])
        print(f"- Config: `{c['file']}` (sha {c['sha256']}); {shown}; overrides: {', '.join(r['overrides']) or 'none'}")
        print(f"- Environment: " + ", ".join(f"`{k}={v}`" for k, v in sorted(r["environment"].items())))
        print()
    print("## Results\n")
    print("| Model | Prompt | Prefill tok/s | Decode avg tok/s (range) | Decode peak tok/s | Second-request setup | Decode rate incl. setup tok/s |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for r in recs:
        for name, v in r["summary"]["by_prompt"].items():
            setup = f"{v['second_setup_s']} s" if v["second_setup_s"] is not None else "-"
            print(f"| {r['name']} | {v['prompt_tokens']} tok | {v['prefill_tok_s']} | {v['decode_avg_tok_s']} "
                  f"({v['decode_min']}-{v['decode_max']}) | {v['decode_peak_tok_s']} | {setup} | {v['request_rate_tok_s']} |")
    print("\n| Model | Prefill average tok/s | Long decode | Speed per bucket (tok/s) | Last bucket vs first |")
    print("| --- | ---: | ---: | --- | ---: |")
    for r in recs:
        l = r["summary"]["long_decode"]
        if l:
            print(f"| {r['name']} | {r['summary']['prefill_avg_tok_s']} | {l['tokens']} tokens, avg {l['avg_tok_s']} | "
                  f"{' / '.join(str(x) for x in l['bucket_tok_s'])} (per {l['bucket_tokens']} tokens) | {l['drift_pct']}% |")
    print("\nLong decode: greedy output is the same on every run, so bucket-to-bucket variation reflects the generated text "
          "(which experts it routes to) as well as the growing context. Read the spread, not a trend line. The prefill average "
          "is the mean over the three prompt lengths; compare models by the per-length numbers, since a 28-token prompt "
          "takes a different (decode-style) path.")
    print("\nMethod: median of repeated runs on a quiet machine; decode = 1000 / mean(iter_total ms) over steps >= 3; "
          "peak = best 32-step window; second-request setup = decode-graph setup after the prefill of a second request.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--model", required=True, help="model file or shard directory (replaces model_path in a temporary config)")
    r.add_argument("--binary", default=os.path.join(ROOT, "build/zeromoe"))
    r.add_argument("--binary-commit", help="commit the binary was built from (default: HEAD if the binary is newer than HEAD)")
    r.add_argument("--env", action="append", default=[], metavar="K=V", help="environment variable for the engine")
    r.add_argument("--set", action="append", default=[], metavar="key=value", help="override a top-level config key")
    r.add_argument("--note", action="append", default=[], metavar="k=v", help="extra setup facts, e.g. ram='DDR4-2666 4ch'")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--long-ngen", type=int, default=1500, help="tokens for the long-decode stability run (0 = skip)")
    r.add_argument("--long-reps", type=int, default=2)
    r.add_argument("--out", default=os.path.join(ROOT, "bench/results"))
    r.add_argument("--deadline", type=float, default=540, help="seconds before pausing (rerun to continue)")
    rr = sub.add_parser("render")
    rr.add_argument("files", nargs="+")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_render(a)


if __name__ == "__main__":
    main()
