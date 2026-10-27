#!/usr/bin/env python3
"""Hardware gate for graph-native Qwen4Exp MTP: speculation must not change output.

Runs the same prompt greedily (temperature 0) with plain decoding and with
graph-native MTP at several draft lengths, then requires byte-identical text.
Verified speculation is exact, so any difference is a bug in the target
verification, state rollback or draft rebuild. Acceptance varies with the
draft length, so identical text across lengths also shows the output does not
depend on how often state was rolled back.

  tools/check_mtp_graph_equivalence.py --binary build/zeromoe \\
      --config configs/qwen38-flash.toml --mtp-model mtp-...-shared-Q4_K_M.gguf \\
      [--model-path ...] [--lengths 1 3 5] [--tokens 48] [--preload-layers 18]

The draft needs ~2 GiB of device memory next to the target's decode graph, so
the target must be configured with that much free VRAM (fewer device expert
layers: --preload-layers). Compare plain and speculative runs at the same
setting; the script uses the same one for both.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile


def with_keys(text, updates, drop_prefixes):
    lines = [l for l in text.splitlines()
             if not any(l.startswith(p) for p in drop_prefixes)]
    # Root keys must precede the first [[gpus]] table.
    cut = next((i for i, l in enumerate(lines) if l.lstrip().startswith("[[")), len(lines))
    return "\n".join(lines[:cut] + updates + [""] + lines[cut:]) + "\n"


def run(binary, cfg_text, name, workdir, prompt, timeout, preload_layers):
    cfg = os.path.join(workdir, name + ".toml")
    log = os.path.join(workdir, name + ".engine.log")
    out = os.path.join(workdir, name + ".out")
    with open(cfg, "w") as f:
        f.write(re.sub(r'(?m)^log_path.*', f'log_path = "{log}"', cfg_text))
    env = os.environ.copy()
    input_text = prompt if prompt.endswith("\n:q\n") else prompt.rstrip("\n") + "\n:q\n"
    with open(out, "w") as o:
        rc = subprocess.run([binary, "--graph-device-expert-layers", str(preload_layers),
                        "--graph-device-expert-trailing", "1", "--config", cfg], input=input_text, text=True,
                            stdout=o, stderr=subprocess.STDOUT, env=env,
                            timeout=timeout).returncode
    text = open(out, errors="replace").read()
    m = re.search(r"Output: (.*?)\n\[Generation Complete\]", text, re.S)
    stats = None
    if os.path.exists(log):
        s = re.search(r"\[MTP-GRAPH\] cycles=(\d+) drafted=(\d+) accepted=(\d+).*restores=(\d+)",
                      open(log, errors="replace").read())
        if s:
            stats = tuple(int(x) for x in s.groups())
    return rc, (m.group(1) if m else None), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--config", required=True, help="a working full-graph Qwen4Exp TOML")
    ap.add_argument("--mtp-model", required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--lengths", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--preload-layers", type=int, default=18)
    ap.add_argument("--prompt", default="Tell a story.\n:q\n")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--keep", help="directory to keep configs/logs in")
    a = ap.parse_args()

    base = open(a.config).read()
    updates = [f"context_size = {a.context}", f"max_new_tokens = {a.tokens}",
               "temperature = 0.0"]
    if a.model_path:
        updates.append(f'model_path = "{a.model_path}"')
    drops = ["context_size", "max_new_tokens", "temperature", "mtp_"]
    if a.model_path:
        drops.append("model_path")
    plain_cfg = with_keys(base, updates, drops)

    workdir = a.keep or tempfile.mkdtemp(prefix="mtp-equiv-")
    os.makedirs(workdir, exist_ok=True)

    results = {}
    rc, text, _ = run(a.binary, plain_cfg, "plain", workdir, a.prompt, a.timeout, a.preload_layers)
    if rc != 0 or not text:
        sys.exit(f"plain run failed (rc={rc}); see {workdir}")
    results["plain"] = text
    ok = True
    for k in a.lengths:
        cfg = with_keys(base, updates + [
            'spec_type = "draft-mtp"', "mtp_graph = true",
            f"spec_draft_n_max = {k}",
            f'spec_draft_model = "{a.mtp_model}"'], drops)
        rc, text, stats = run(a.binary, cfg, f"k{k}", workdir, a.prompt, a.timeout, a.preload_layers)
        if rc != 0 or not text or not stats:
            print(f"FAIL k={k}: rc={rc}, output={'yes' if text else 'no'}, "
                  f"MTP-GRAPH stats={'yes' if stats else 'no'} ({workdir})")
            ok = False
            continue
        cycles, drafted, accepted, restores = stats
        same = text == results["plain"]
        ok = ok and same
        print(f"{'ok  ' if same else 'FAIL'} k={k}: identical to plain={same}, "
              f"drafted={drafted} accepted={accepted} "
              f"({100.0 * accepted / drafted if drafted else 0:.1f}%), "
              f"tokens/cycle={a.tokens / cycles:.2f}, restores={restores}")
        if not same:
            i = next((i for i, (x, y) in enumerate(zip(text, results["plain"])) if x != y),
                     min(len(text), len(results["plain"])))
            print(f"     first difference at char {i}: plain={results['plain'][i-20:i+40]!r} "
                  f"spec={text[i-20:i+40]!r}")
    print("PASS" if ok else "FAIL", f"(artifacts: {workdir})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
