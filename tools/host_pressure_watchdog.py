#!/usr/bin/env python3
"""Stop a guarded model run before swap pressure can stall the host.

The watchdog is deliberately separate from request handling: an HTTP/model
failure leaves ZeroMoE running, while host-memory safety violations send
SIGTERM and escalate to SIGKILL if the same process remains after a short
grace period. It never signals a PID whose /proc start time changed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import time
import tomllib

GIB = 1024 ** 3
MIB = 1024 ** 2


def parse_meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        fields = value.split()
        if fields:
            result[key] = int(fields[0]) * 1024
    return result


def parse_status(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep and key in {"VmRSS", "VmSwap", "VmLck"}:
            result[key] = int(value.split()[0]) * 1024
    return result


def proc_start_ticks(pid: int, proc_root: Path = Path("/proc")) -> int:
    # comm may contain spaces and parentheses, so parse after its final ')'.
    stat = (proc_root / str(pid) / "stat").read_text()
    tail = stat[stat.rfind(")") + 1 :].split()
    return int(tail[19])  # field 22 (starttime), with field 3 at tail[0]


def pressure_reasons(
    *,
    memavailable_gib: float,
    vm_swap_gib: float,
    swap_used_gib: float,
    baseline_swap_gib: float,
    psi_full_avg10_pct: float,
    full_psi_streak: int,
    swapout_mib_s: float,
    swapout_streak: int,
    live_floor_gib: float,
    # The previous DeepSeek run escalated to 39+ GiB system swap and froze
    # the host. Stop on early signs of new swapping instead of waiting for
    # severe pressure.
    process_swap_limit_gib: float = 0.0625,  # 64 MiB
    system_swap_growth_limit_gib: float = 0.125,  # 128 MiB
    psi_full_limit_pct: float = 1.0,
    psi_full_samples: int = 1,
    swapout_limit_mib_s: float = 1.0,
    swapout_samples: int = 1,
) -> list[str]:
    """Pure policy function, also used by the synthetic safety test."""
    reasons: list[str] = []
    if memavailable_gib < live_floor_gib:
        reasons.append("MemAvailable below configured live floor")
    if vm_swap_gib > process_swap_limit_gib:
        reasons.append(f"ZeroMoE VmSwap exceeded {process_swap_limit_gib:.2f} GiB")
    if swap_used_gib - baseline_swap_gib > system_swap_growth_limit_gib:
        reasons.append(
            "system SwapUsed grew more than "
            f"{system_swap_growth_limit_gib:.2f} GiB from launch baseline"
        )
    if (psi_full_avg10_pct > psi_full_limit_pct and
            full_psi_streak >= psi_full_samples):
        reasons.append(
            "memory PSI full avg10 exceeded "
            f"{psi_full_limit_pct:.1f}% for consecutive samples"
        )
    if (swapout_mib_s > swapout_limit_mib_s and
            swapout_streak >= swapout_samples):
        reasons.append(
            "swap-out exceeded "
            f"{swapout_limit_mib_s:.1f} MiB/s for consecutive samples"
        )
    return reasons


def read_psi_avg10(kind: str, path: Path = Path("/proc/pressure/memory")) -> float:
    line = next(x for x in path.read_text().splitlines()
                if x.startswith(kind + " "))
    return float(next(x.split("=", 1)[1] for x in line.split()
                      if x.startswith("avg10=")))


def read_pswpout(path: Path = Path("/proc/vmstat")) -> int:
    line = next(x for x in path.read_text().splitlines()
                if x.startswith("pswpout "))
    return int(line.split()[1])


def server_pid(pid_file: Path) -> int | None:
    try:
        for line in pid_file.read_text().splitlines():
            if line.startswith("server_pid="):
                return int(line.split("=", 1)[1])
    except OSError:
        return None
    return None


def swap_baseline_from_gate(pid_file: Path) -> float | None:
    try:
        for line in pid_file.read_text().splitlines():
            if line.startswith("swap_baseline_gib="):
                return float(line.split("=", 1)[1])
    except OSError:
        pass
    return None


def watchdog_live_floor_gib(config: dict[str, object]) -> float:
    """Read the watchdog floor, separate from the engine request floor."""
    floor_mb = float(config.get(
        "host_pressure_watchdog_floor_mb",
        config.get("graph_mem_guard_floor_mb", 4608),
    ))
    if not math.isfinite(floor_mb) or floor_mb <= 0:
        raise ValueError("host pressure watchdog floor must be a positive finite MiB value")
    return floor_mb / 1024.0


def terminate_for_host_safety(
    pid: int,
    expected_start_ticks: int,
    proc_root: Path = Path("/proc"),
    grace_seconds: float = 2.0,
    poll_interval: float = 0.1,
) -> str:
    """Stop the guarded server on host pressure, escalating only if needed."""
    try:
        if proc_start_ticks(pid, proc_root) != expected_start_ticks:
            return "PID identity changed; no signal sent"
    except OSError:
        return "process exited before SIGTERM"

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "process exited before SIGTERM"

    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        try:
            current_start_ticks = proc_start_ticks(pid, proc_root)
        except OSError:
            return "SIGTERM sent; process exited during grace period"
        if current_start_ticks != expected_start_ticks:
            return "SIGTERM sent; PID identity changed during grace period"
        remaining = max(0.0, deadline - time.monotonic())
        if remaining == 0.0:
            break
        time.sleep(min(max(0.001, poll_interval), remaining))

    try:
        current_start_ticks = proc_start_ticks(pid, proc_root)
    except OSError:
        return "SIGTERM sent; process exited during grace period"
    if current_start_ticks != expected_start_ticks:
        return "SIGTERM sent; PID identity changed during grace period"
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "SIGTERM sent; process exited before SIGKILL"
    return "SIGTERM sent; SIGKILL sent after grace period"


def run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    pid_file = args.pid_file.resolve()
    config = tomllib.loads(args.config.resolve().read_text())
    live_floor_gib = watchdog_live_floor_gib(config)
    proc_root = Path(args.proc_root)
    page_size = os.sysconf("SC_PAGE_SIZE")
    started = time.monotonic()
    baseline_swap_gib = swap_baseline_from_gate(pid_file)
    expected_start_ticks: int | None = None
    previous_pages: int | None = None
    previous_mono: float | None = None
    full_psi_streak = 0
    swapout_streak = 0
    output = args.output.resolve()
    emergency_path = args.emergency_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", buffering=1) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "utc", "elapsed_s", "pid", "MemAvailable_GiB", "SwapUsed_GiB",
            "SwapGrowth_GiB", "VmRSS_GiB", "VmSwap_GiB", "VmLck_GiB",
            "psi_some_avg10", "psi_full_avg10", "swapout_MiB_s",
            "emergency_stop", "emergency_reason",
        ])
        while True:
            pid = server_pid(pid_file)
            if pid is None:
                return 0
            proc_dir = proc_root / str(pid)
            if not proc_dir.exists():
                return 0
            try:
                start_ticks = proc_start_ticks(pid, proc_root)
                if expected_start_ticks is None:
                    expected_start_ticks = start_ticks
                elif start_ticks != expected_start_ticks:
                    # PID was reused; never signal the unrelated process.
                    return 0
                status = parse_status(proc_dir / "status")
                meminfo = parse_meminfo(proc_root / "meminfo")
                available = meminfo["MemAvailable"]
                swap_used = max(0, meminfo["SwapTotal"] - meminfo["SwapFree"])
                swap_used_gib = swap_used / GIB
                if baseline_swap_gib is None:
                    baseline_swap_gib = swap_used_gib
                swap_growth_gib = max(0.0, swap_used_gib - baseline_swap_gib)
                psi_some = read_psi_avg10("some", proc_root / "pressure/memory")
                psi_full = read_psi_avg10("full", proc_root / "pressure/memory")
                now = time.monotonic()
                pages = read_pswpout(proc_root / "vmstat")
                swapout_mib_s = 0.0
                if previous_pages is not None and previous_mono is not None:
                    elapsed = max(0.001, now - previous_mono)
                    swapout_mib_s = max(0, pages - previous_pages) * page_size / MIB / elapsed
                previous_pages, previous_mono = pages, now
                full_psi_streak = full_psi_streak + 1 if psi_full > 1.0 else 0
                swapout_streak = swapout_streak + 1 if swapout_mib_s > 1.0 else 0
                sample = {
                    "memavailable_gib": available / GIB,
                    "vm_swap_gib": status.get("VmSwap", 0) / GIB,
                    "swap_used_gib": swap_used_gib,
                    "psi_full_avg10_pct": psi_full,
                    "swapout_mib_s": swapout_mib_s,
                }
                reasons = pressure_reasons(
                    **sample,
                    baseline_swap_gib=baseline_swap_gib,
                    full_psi_streak=full_psi_streak,
                    swapout_streak=swapout_streak,
                    live_floor_gib=live_floor_gib,
                )
                writer.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    round(now - started, 1), pid,
                    round(sample["memavailable_gib"], 3),
                    round(swap_used_gib, 3), round(swap_growth_gib, 3),
                    round(status.get("VmRSS", 0) / GIB, 3),
                    round(sample["vm_swap_gib"], 3),
                    round(status.get("VmLck", 0) / GIB, 3),
                    psi_some, psi_full, round(swapout_mib_s, 3),
                    bool(reasons), "; ".join(reasons),
                ])
                if reasons:
                    emergency = {
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "pid": pid,
                        "reasons": reasons,
                        "sample": sample,
                        "baseline_swap_used_gib": baseline_swap_gib,
                        "action": "host-safety termination requested",
                    }
                    emergency_path.write_text(json.dumps(emergency, indent=2) + "\n")
                    # Request failures alone never signal ZeroMoE. This branch
                    # is only reachable on explicit host-memory pressure.
                    emergency["action"] = terminate_for_host_safety(
                        pid,
                        expected_start_ticks,
                        proc_root=proc_root,
                        grace_seconds=args.terminate_grace_seconds,
                    )
                    emergency_path.write_text(json.dumps(emergency, indent=2) + "\n")
                    return 2
            except (KeyError, OSError, StopIteration, ValueError) as exc:
                with (run_dir / "host-monitor-errors.log").open("a") as errors:
                    errors.write(f"{datetime.now(timezone.utc).isoformat()} {exc!r}\n")
            time.sleep(args.interval)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emergency-output", type=Path, required=True)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--terminate-grace-seconds", type=float, default=2.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
