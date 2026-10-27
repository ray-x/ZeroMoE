#!/usr/bin/env python3
"""Collect one standard performance record per model/config on this machine, and render records as Markdown.

  bench/perf_report.py run --name qwen38-flash --config configs/qwen38-flash.toml \\
      --model /path/to/model --binary build/zeromoe \\
      --set omp_max_threads=10 --note ram="DDR4-2666 4ch"
  bench/perf_report.py render bench/results/*.json

The measurements come from the engine itself: this script starts `zeromoe --bench` (core/bench_mode.hpp), which loads
the model once and runs every request through the normal request path, and reads the per-request records the engine
writes (core/request_stats.hpp). It does not parse log lines. What it adds is what only an outside observer can do:
describe the machine and the build, and decide whether the machine was quiet while each request ran.

A record holds
  * setup: CPU (AVX-512/VNNI/VBMI), RAM, GPUs (PCIe link, power limit), OS, CUDA, ZeroMoE commit and whether the binary
    matches it, ggml commit and whether its sources were modified, model files, the config (model path redacted),
    every override and environment variable, and the engine's resolved-settings log lines;
  * real text (bench/prompts/short, mid, long): prefill tok/s per prompt length and their average, steady decode tok/s,
    best 32-step window, time to first token, decode-graph setup, and the decode rate including that setup;
  * long decode: one long generation split into six equal parts, to show how speed moves over a generation;
  * random tokens, as llama-bench does (pp512 and tg128 by default, fixed seed): `llama-bench -p 512 -n 128`
    equivalent. Random ids route to random experts, so a MoE model is slower on them than on real text; both are
    reported.

A request counts only if the engine marked it valid (full token count, no prompt-cache hit) and the machine was quiet
during it: outside CPU (everything except the engine) at most 0.5 cores on average and 1.0 in any second during the
request's prefill and its decode, no other zeromoe, no compiler. Requests that fail are kept under "rejected" and
replaced by further runs (three attempts per mode). Speeds are medians of the accepted requests, with min and max.
"""
import argparse, glob, hashlib, json, os, platform, re, statistics as st, subprocess, sys, time

HZ = os.sysconf("SC_CLK_TCK")
NOISY = {"cicc", "cc1plus", "cc1", "nvcc", "ptxas", "fatbinary", "ld", "collect2", "ninja", "make", "cmake", "rsync"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = {"short": "bench/prompts/short.txt", "mid": "bench/prompts/mid.txt", "long": "bench/prompts/long.txt"}
STD_NGEN = 200
SCHEMA = 2
MAX_ATTEMPTS = 3


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
    head = git("rev-parse --short=12 HEAD")
    built_from = binary_commit or (head if newer else "unknown (binary is older than HEAD; pass --binary-commit)")
    return {
        "zeromoe_commit": built_from,
        "repo_head": head,
        "zeromoe_tree_dirty": bool(git("status --porcelain --untracked-files=no")) if built_from == head else None,
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


def set_config_key(cfg, key, value):
    """Set a top-level TOML key; a new key goes before the first [[table]] (keys after it belong to that table)."""
    pat = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
    if pat.search(cfg):
        return pat.sub(f"{key} = {value}", cfg, count=1)
    i = cfg.find("\n[[")
    return cfg + f"\n{key} = {value}\n" if i < 0 else cfg[:i] + f"\n{key} = {value}\n" + cfg[i:]


# ---------------------------------------------------------------- quiet-machine handling
def busy_ticks():
    v = list(map(int, open("/proc/stat").readline().split()[1:]))
    return v[0] + v[1] + v[2] + v[5] + v[6] + v[7]  # user nice system irq softirq steal


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


# ---------------------------------------------------------------- one bench process
def run_bench(args, tag, bench_args, cfg_text, env_extra):
    """Starts `zeromoe --bench ...`, samples the machine every second while it runs, returns (bench json, samples, events)."""
    logdir = os.path.join(args.out, f"{args.name}.logs")
    os.makedirs(logdir, exist_ok=True)
    cfg = re.sub(r"(?m)^model_path\s*=.*$", f'model_path = "{args.model}"', cfg_text)
    cfg = re.sub(r"(?m)^log_path\s*=.*$", f'log_path = "{logdir}/{tag}.engine.log"', cfg)
    for kv in args.set:
        k, _, v = kv.partition("=")
        cfg = set_config_key(cfg, k, v)
    cfg_file = f"{logdir}/{tag}.toml"
    open(cfg_file, "w").write(cfg)
    result_file, stats_file = f"{logdir}/{tag}.bench.json", f"{logdir}/{tag}.stats.jsonl"
    for f in (result_file, stats_file):
        if os.path.exists(f):
            os.remove(f)
    env = dict(os.environ, **env_extra)
    cmd = [args.binary, "--config", cfg_file, "--bench", "--bench-out", result_file, "--stats-file", stats_file] + bench_args
    out = open(f"{logdir}/{tag}.out", "w")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
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
    if p.returncode != 0 or not os.path.exists(result_file):
        tail = open(f"{logdir}/{tag}.out", errors="replace").read()[-600:]
        print(f"{tag}: engine exited {p.returncode}, no result.\n{tail}")
        return None, samples, events
    return json.load(open(result_file)), samples, events


def window_quiet(samples, events, t0_ms, t1_ms):
    """(quiet?, mean outside cores, max outside cores) over [t0_ms, t1_ms] wall-clock."""
    a, b = t0_ms / 1000.0, t1_ms / 1000.0
    ws = [e for (t, e) in samples if t > a and t - 1.0 < b]
    ev = [x for x in events if a - 1.0 <= x[0] <= b + 1.0]
    if not ws:
        return False, None, None
    mean, mx = sum(ws) / len(ws), max(ws)
    return (mean <= 0.5 and mx <= 1.0 and not ev), round(mean, 2), round(mx, 2)


def screen(requests, samples, events):
    """Annotate every measured request with a quiet verdict for its prefill and decode windows."""
    for r in requests:
        s, d, e = r.get("start_unix_ms"), r.get("decode_start_unix_ms"), r.get("end_unix_ms")
        if s is None or e is None:
            r["quiet"], r["outside_cpu"] = False, None
            continue
        d = d if d is not None else s
        q1, m1, x1 = window_quiet(samples, events, s, d)
        q2, m2, x2 = window_quiet(samples, events, d, e)
        r["quiet"] = q1 and q2
        r["outside_cpu"] = {"prefill": [m1, x1], "decode": [m2, x2]}
    return requests


# ---------------------------------------------------------------- aggregation
def spread(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"median": round(st.median(xs), 2), "min": round(min(xs), 2), "max": round(max(xs), 2)}


def aggregate_test(rs):
    return {
        "n": len(rs), "prompt_tokens": rs[0].get("prompt_tokens"),
        "prefill_tok_s": spread([r.get("prefill_tok_s") for r in rs]),
        "decode_tok_s": spread([r.get("decode_tok_s") for r in rs]),
        "decode_peak_tok_s": spread([r.get("decode_peak_tok_s") for r in rs]),
        "decode_p50_ms": spread([r.get("decode_p50_ms") for r in rs]),
        "decode_p95_ms": spread([r.get("decode_p95_ms") for r in rs]),
        "ttft_ms": spread([r.get("ttft_ms") for r in rs]),
        "decode_setup_ms": spread([r.get("decode_setup_ms") for r in rs]),
        "decode_wall_tok_s": spread([r.get("decode_wall_tok_s") for r in rs]),
    }


def summarize(accepted):
    by = {}
    for r in accepted:
        by.setdefault((r["mode"], r["kind"], r["label"]), []).append(r)
    text = {"by_prompt": {}}
    for name in PROMPTS:
        rs = by.get(("text", "test", name))
        if rs:
            text["by_prompt"][name] = aggregate_test(rs)
    pre = [v["prefill_tok_s"]["median"] for v in text["by_prompt"].values() if v["prefill_tok_s"]]
    text["prefill_avg_tok_s"] = round(st.mean(pre), 1) if pre else None
    lr = by.get(("text", "long", "long_decode"))
    if lr:
        bk = [r["decode_bucket_tok_s"] for r in lr if r.get("decode_bucket_tok_s")]
        per = [round(st.median(b[i] for b in bk), 2) for i in range(6)] if bk else None
        text["long_decode"] = {"tokens": lr[0]["generated_tokens"], "n": len(lr), "decode_tok_s": spread([r.get("decode_tok_s") for r in lr]),
                               "bucket_tok_s": per, "bucket_tokens": (lr[0]["generated_tokens"] - 3) // 6 if per else None,
                               "last_vs_first_pct": round((per[-1] / per[0] - 1) * 100, 1) if per else None}
    random = {}
    for (mode, kind, label), rs in sorted(by.items()):
        if mode == "random" and kind == "test":
            random[label] = aggregate_test(rs)
    return {"text": text, "random": random}


# ---------------------------------------------------------------- driver
def cmd_run(args):
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    cfg_text = open(cfg_path).read()
    env_extra = dict(kv.split("=", 1) for kv in args.env)
    os.makedirs(args.out, exist_ok=True)
    accepted_f = os.path.join(args.out, f"{args.name}.runs.jsonl")
    rejected_f = os.path.join(args.out, f"{args.name}.rejected.jsonl")
    accepted = [json.loads(l) for l in open(accepted_f)] if os.path.exists(accepted_f) else []
    t_end = time.time() + args.deadline

    def count(mode, kind, label):
        return sum(1 for r in accepted if (r["mode"], r["kind"], r["label"]) == (mode, kind, label))

    targets = [("text", "test", n, args.reps) for n in PROMPTS]
    if args.long_ngen:
        targets.append(("text", "long", "long_decode", args.long_reps))
    if not args.no_random:
        targets += [("random", "test", f"pp{n}", args.reps) for n in args.random_pp]
        targets += [("random", "test", f"tg{n}", args.reps) for n in args.random_tg]

    for mode in ("text", "random"):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            need = {(k, l): t - count(m, k, l) for (m, k, l, t) in targets if m == mode and t - count(m, k, l) > 0}
            if not need:
                break
            if time.time() > t_end - 360:
                print(f"time budget reached before the {mode} measurements; rerun the same command to continue")
                return
            if not wait_quiet():
                print("machine never became quiet; stopping")
                return
            reps = max(need.values())
            if mode == "text":
                prompts = [l for (k, l) in need if k == "test"] or ["short"]
                bargs = ["--bench-ngen", str(STD_NGEN), "--bench-reps", str(reps), "--bench-warmup", "1"]
                for n in prompts:
                    bargs += ["--bench-prompt", os.path.join(ROOT, PROMPTS[n])]
                if ("long", "long_decode") in need:
                    bargs += ["--bench-long", str(args.long_ngen), "--bench-long-reps", str(need[("long", "long_decode")])]
            else:
                bargs = ["--bench-reps", str(reps), "--bench-warmup", "1", "--bench-seed", str(args.seed)]
                for (k, l) in need:
                    bargs += ["--bench-random-pp" if l.startswith("pp") else "--bench-random-tg", l[2:]]
            tag = f"{mode}_a{attempt}_{len(accepted)}"
            print(f"[{mode}] attempt {attempt}: need {dict((l, n) for (k, l), n in need.items())}", flush=True)
            res, samples, events = run_bench(args, tag, bargs, cfg_text, env_extra)
            if res is None:
                continue
            reqs = [r for r in res["requests"] if r["kind"] != "warmup"]
            for r in reqs:
                r["mode"] = res.get("mode", mode)
            screen(reqs, samples, events)
            for r in reqs:
                key = (r["mode"], r["kind"], r["label"])
                tgt = next((t for (m, k, l, t) in targets if (m, k, l) == key), 0)
                ok = r["valid"] and r["quiet"] and count(*key) < tgt
                r["attempt"] = attempt
                if ok:
                    accepted.append(r)
                    open(accepted_f, "a").write(json.dumps(r) + "\n")
                elif not (r["valid"] and r["quiet"]):
                    open(rejected_f, "a").write(json.dumps(r) + "\n")
                print(f"  {r['label']:12s} rep {r['rep']}: {'accepted' if ok else ('extra' if r['valid'] and r['quiet'] else 'REJECTED')}"
                      f" valid={r['valid']} quiet={r['quiet']} prefill={r.get('prefill_tok_s')} decode={r.get('decode_tok_s')}"
                      f" outside={r.get('outside_cpu')}", flush=True)
    missing = [(m, l) for (m, k, l, t) in targets if count(m, k, l) < t]
    if missing:
        print("not enough clean requests for:", missing, "- the record is written with what was accepted")
    finalize(args, cfg_path, cfg_text, env_extra, accepted, rejected_f)


def finalize(args, cfg_path, cfg_text, env_extra, accepted, rejected_f):
    logdir = os.path.join(args.out, f"{args.name}.logs")
    first = sorted(glob.glob(os.path.join(logdir, "text_a1_*.engine.log")) + glob.glob(os.path.join(logdir, "random_a1_*.engine.log")))
    log = open(first[0], errors="replace").read() if first else ""
    keep = ("[SETTINGS]", "[GRAPH-CAPS]", "[GRAPH-TUNING]", "device expert arena", "soft expert tier", "[GRAPH-RAM] expert-arena")
    engine_lines = [re.sub(r"^\[[\d:.]+\] ", "", l)[:300] for l in log.splitlines() if any(k in l for k in keep)][:24]
    rejected = sum(1 for _ in open(rejected_f)) if os.path.exists(rejected_f) else 0
    record = {
        "schema": SCHEMA, "name": args.name, "date": time.strftime("%Y-%m-%d"),
        "system": system_info(env_extra.get("CUDA_VISIBLE_DEVICES")), "build": build_info(args.binary, args.binary_commit),
        "model": model_info(args.model), "config": config_info(cfg_path, cfg_text), "overrides": args.set,
        "environment": {**{k: v for k, v in os.environ.items() if re.match(r"(ZEROMOE_|CUDA_VISIBLE|OMP_|GGML_)", k)}, **env_extra},
        "notes": dict(kv.split("=", 1) for kv in args.note), "engine_log_lines": engine_lines,
        "protocol": {
            "source": "zeromoe --bench per-request records (core/request_stats.hpp)", "standard_ngen": STD_NGEN, "reps": args.reps,
            "prompts": PROMPTS, "random_seed": args.seed, "random_tests": [] if args.no_random else
            [f"pp{n}" for n in args.random_pp] + [f"tg{n}" for n in args.random_tg],
            "decode_definition": "1000 / mean(step ms) over steps >= 3", "peak_definition": "best 32-step window",
            "quiet_rule": "outside CPU <= 0.5 cores mean and <= 1.0 in any second during prefill and decode; no other zeromoe or compiler",
            "accepted": len(accepted), "rejected": rejected},
        "summary": summarize(accepted),
        "runs": accepted,
    }
    path = os.path.join(args.out, f"{args.name}.json")
    json.dump(record, open(path, "w"), indent=1)
    print("wrote", path)


# ---------------------------------------------------------------- render
def med(sp):
    return None if not sp else sp["median"]


def rng(sp, d=1):
    return "-" if not sp else f"{sp['median']:.{d}f} ({sp['min']:.{d}f}-{sp['max']:.{d}f})"


def num(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def normalize(r):
    """Schema 1 records (log-scraped, one process per run) rendered with the same code."""
    if r.get("schema", 1) >= 2:
        return r
    old = r["summary"]
    sp = lambda lo, mid, hi: {"median": mid, "min": lo, "max": hi}
    by = {}
    for name, v in old["by_prompt"].items():
        by[name] = {"n": v["n"], "prompt_tokens": v["prompt_tokens"], "prefill_tok_s": sp(v["prefill_min"], v["prefill_tok_s"], v["prefill_max"]),
                    "decode_tok_s": sp(v["decode_min"], v["decode_avg_tok_s"], v["decode_max"]), "decode_peak_tok_s": sp(v["decode_peak_tok_s"], v["decode_peak_tok_s"], v["decode_peak_tok_s"]),
                    "ttft_ms": None, "decode_setup_ms": {"median": (v["second_setup_s"] or 0) * 1000, "min": 0, "max": 0} if v["second_setup_s"] is not None else None,
                    "decode_wall_tok_s": sp(v["request_rate_tok_s"], v["request_rate_tok_s"], v["request_rate_tok_s"])}
    ld = old.get("long_decode")
    text = {"by_prompt": by, "prefill_avg_tok_s": old["prefill_avg_tok_s"]}
    if ld:
        text["long_decode"] = {"tokens": ld["tokens"], "n": ld["n"], "decode_tok_s": sp(ld["avg_tok_s"], ld["avg_tok_s"], ld["avg_tok_s"]),
                               "bucket_tok_s": ld["bucket_tok_s"], "bucket_tokens": ld["bucket_tokens"], "last_vs_first_pct": ld["drift_pct"]}
    r = dict(r)
    r["summary"] = {"text": text, "random": {}}
    return r


def cmd_render(args):
    recs = [normalize(json.load(open(f))) for f in args.files]
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
    print("## Results: real text\n")
    print("| Model | Prompt | Prefill tok/s | Decode tok/s (range) | Decode peak tok/s | Time to first token | Second-request setup | Decode incl. setup tok/s |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in recs:
        for name, v in r["summary"]["text"]["by_prompt"].items():
            setup = f"{med(v['decode_setup_ms'])/1000:.2f} s" if v.get("decode_setup_ms") else "-"
            ttft = f"{med(v['ttft_ms'])/1000:.2f} s" if v.get("ttft_ms") else "-"
            print(f"| {r['name']} | {v['prompt_tokens']} tok | {num(med(v['prefill_tok_s']))} | {rng(v['decode_tok_s'])} | "
                  f"{num(med(v['decode_peak_tok_s']))} | {ttft} | {setup} | {num(med(v['decode_wall_tok_s']))} |")
    print("\n| Model | Prefill average tok/s | Long decode | Speed per part (tok/s) | Last part vs first |")
    print("| --- | ---: | --- | --- | ---: |")
    for r in recs:
        t = r["summary"]["text"]
        l = t.get("long_decode")
        if l:
            parts = " / ".join(str(x) for x in l["bucket_tok_s"]) if l["bucket_tok_s"] else "-"
            print(f"| {r['name']} | {t['prefill_avg_tok_s']} | {l['tokens']} tokens, avg {num(med(l['decode_tok_s']), 2)} | {parts} (per {l['bucket_tokens']} tokens) | {num(l['last_vs_first_pct'])}% |")
    if any(r["summary"]["random"] for r in recs):
        print("\n## Results: random tokens (llama-bench style: `llama-bench -p N -n N` equivalent, fixed seed)\n")
        print("| Model | Test | Tokens/s (range) | Peak tok/s |")
        print("| --- | --- | ---: | ---: |")
        for r in recs:
            for label, v in r["summary"]["random"].items():
                sp = v["prefill_tok_s"] if label.startswith("pp") else v["decode_tok_s"]
                pk = "-" if label.startswith("pp") else num(med(v["decode_peak_tok_s"]))
                print(f"| {r['name']} | {label} | {rng(sp)} | {pk} |")
        print("\nRandom token ids route to random experts, which a hot-expert placement tuned on real text does not favour: "
              "expect decode well below the real-text figures for a MoE model. `tg` starts from a one-token prompt and feeds "
              "a random token at every step instead of the sampled one, as llama-bench does.")
    print("\nLong decode: greedy output is identical on every run, so variation between parts reflects what is generated (which "
          "experts it routes to) as well as the growing context; read the spread, not a trend line. Prefill average is the mean "
          "over the three prompt lengths; compare models by the per-length numbers.")
    print("\nMethod: `zeromoe --bench` (one process, requests through the normal request path); median of accepted requests, "
          "a request counts only if it produced all its tokens and the machine was quiet during it; decode = 1000 / mean(step ms) over "
          "steps >= 3; peak = best 32-step window; second-request setup = decode-graph setup paid by a request after its prefill.")


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
    r.add_argument("--long-ngen", type=int, default=1500, help="tokens for the long-decode run (0 = skip)")
    r.add_argument("--long-reps", type=int, default=2)
    r.add_argument("--random-pp", type=int, action="append", default=None, metavar="N", help="random-token prompt sizes (default 512)")
    r.add_argument("--random-tg", type=int, action="append", default=None, metavar="N", help="random-token generation lengths (default 128)")
    r.add_argument("--no-random", action="store_true", help="skip the random-token tests")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", default=os.path.join(ROOT, "bench/results"))
    r.add_argument("--deadline", type=float, default=3000, help="seconds before pausing (rerun to continue)")
    rr = sub.add_parser("render")
    rr.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.cmd == "run":
        a.random_pp = a.random_pp or [512]
        a.random_tg = a.random_tg or [128]
        cmd_run(a)
    else:
        cmd_render(a)


if __name__ == "__main__":
    main()
