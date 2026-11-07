#!/usr/bin/env python3
"""Synthetic checks for the guarded inference host-swap watchdog."""

import signal
from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "host_pressure_watchdog.py"
SPEC = importlib.util.spec_from_file_location("host_pressure_watchdog", MODULE_PATH)
WATCHDOG = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(WATCHDOG)


class ProcessStatusParsingTests(unittest.TestCase):
    def test_reads_pinned_and_locked_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / "status"
            status.write_text(
                "VmRSS: 1024 kB\nVmSwap: 8 kB\nVmLck: 16 kB\nVmPin: 32 kB\n"
            )
            self.assertEqual(
                WATCHDOG.parse_status(status),
                {"VmRSS": 1024 * 1024, "VmSwap": 8 * 1024,
                 "VmLck": 16 * 1024, "VmPin": 32 * 1024},
            )


class PressureReasonTests(unittest.TestCase):
    def reasons(self, **overrides):
        values = {
            "memavailable_gib": 5.5,
            "vm_swap_gib": 0.0,
            "swap_used_gib": 1.317,
            "baseline_swap_gib": 1.317,
            "psi_full_avg10_pct": 1.0,
            "full_psi_streak": 0,
            "psi_some_avg10_pct": 1.0,
            "some_psi_streak": 0,
            "swapout_mib_s": 0.0,
            "swapout_streak": 0,
            "live_floor_gib": 4.5,
        }
        values.update(overrides)
        return WATCHDOG.pressure_reasons(**values)

    def test_prior_healthy_deepseek_run_sample_is_allowed(self):
        self.assertEqual(self.reasons(), [])

    def test_v26_freeze_sample_triggers_host_safety(self):
        reasons = self.reasons(
            memavailable_gib=0.0,
            vm_swap_gib=28.005,
            swap_used_gib=39.625,
            baseline_swap_gib=1.317,
            psi_full_avg10_pct=98.4,
            full_psi_streak=3,
            psi_some_avg10_pct=98.4,
            some_psi_streak=3,
            swapout_mib_s=100.0,
            swapout_streak=3,
        )
        self.assertGreaterEqual(len(reasons), 5)

    def test_small_process_swap_growth_trips_early(self):
        reasons = self.reasons(vm_swap_gib=0.063)
        self.assertTrue(any("VmSwap" in reason for reason in reasons))

    def test_small_system_swap_growth_trips_early(self):
        reasons = self.reasons(swap_used_gib=1.426, baseline_swap_gib=1.300)
        self.assertTrue(any("SwapUsed" in reason for reason in reasons))

    def test_v38_preload_swap_growth_stops_before_request(self):
        reasons = self.reasons(
            memavailable_gib=10.925,
            vm_swap_gib=0.0,
            swap_used_gib=0.2626,
            baseline_swap_gib=0.002,
            psi_full_avg10_pct=0.54,
            full_psi_streak=0,
            psi_some_avg10_pct=0.54,
            some_psi_streak=0,
            swapout_mib_s=243.303,
            swapout_streak=1,
        )
        self.assertTrue(any("SwapUsed grew" in reason for reason in reasons))

    def test_sustained_high_swapout_trips(self):
        reasons = self.reasons(swapout_mib_s=65.0, swapout_streak=2)
        self.assertTrue(any("swap-out" in reason for reason in reasons))

    def test_v37_watchdog_sample_with_ample_headroom_is_allowed(self):
        reasons = self.reasons(
            memavailable_gib=12.625,
            vm_swap_gib=0.0,
            swap_used_gib=0.0017,
            baseline_swap_gib=0.001,
            psi_full_avg10_pct=1.01,
            full_psi_streak=1,
            psi_some_avg10_pct=1.01,
            some_psi_streak=1,
            swapout_mib_s=0.226,
            swapout_streak=0,
        )
        self.assertEqual(reasons, [])

    def test_psi_full_requires_low_mem_and_two_samples(self):
        reasons = self.reasons(
            memavailable_gib=12.0,
            psi_full_avg10_pct=4.1,
            full_psi_streak=2,
        )
        self.assertTrue(any("PSI full" in reason for reason in reasons))

    def test_psi_some_requires_low_mem_and_two_samples(self):
        reasons = self.reasons(
            memavailable_gib=40.0,
            psi_some_avg10_pct=20.1,
            some_psi_streak=2,
        )
        self.assertTrue(any("PSI some" in reason for reason in reasons))

    def test_process_swap_is_separately_gated(self):
        reasons = self.reasons(vm_swap_gib=0.3)
        self.assertTrue(any("VmSwap" in reason for reason in reasons))

    def test_v26_first_live_floor_crossing_stops_before_large_swap(self):
        reasons = self.reasons(
            memavailable_gib=2.917,
            vm_swap_gib=0.0,
            swap_used_gib=1.317,
            baseline_swap_gib=1.317,
            psi_full_avg10_pct=0.0,
        )
        self.assertIn("MemAvailable below configured live floor", reasons)

    def test_swap_growth_is_gated_before_psi_reaches_critical_level(self):
        reasons = self.reasons(
            memavailable_gib=8.0,
            vm_swap_gib=0.0,
            swap_used_gib=2.086,
            baseline_swap_gib=1.317,
            psi_full_avg10_pct=2.0,
        )
        self.assertTrue(any("SwapUsed grew" in reason for reason in reasons))

    def test_request_failure_without_memory_pressure_does_not_trigger_watchdog(self):
        # HTTP status is intentionally not an input to the host-safety policy.
        self.assertEqual(self.reasons(), [])


class SafetyTerminationTests(unittest.TestCase):
    def test_pid_reuse_is_never_signaled(self):
        with mock.patch.object(WATCHDOG, "proc_start_ticks", return_value=456):
            with mock.patch.object(WATCHDOG.os, "kill") as kill:
                action = WATCHDOG.terminate_for_host_safety(123, 123)
        self.assertIn("identity changed", action)
        kill.assert_not_called()

    def test_escalates_to_sigkill_only_after_grace_for_same_process(self):
        with mock.patch.object(WATCHDOG, "proc_start_ticks", return_value=123):
            with mock.patch.object(WATCHDOG.os, "kill") as kill:
                action = WATCHDOG.terminate_for_host_safety(
                    123, 123, grace_seconds=0.0
                )
        self.assertIn("SIGKILL sent after grace period", action)
        self.assertEqual(
            [call.args[1] for call in kill.call_args_list],
            [signal.SIGTERM, signal.SIGKILL],
        )

    def test_does_not_escalate_if_process_exits_after_sigterm(self):
        with mock.patch.object(
            WATCHDOG,
            "proc_start_ticks",
            side_effect=[123, FileNotFoundError],
        ):
            with mock.patch.object(WATCHDOG.os, "kill") as kill:
                action = WATCHDOG.terminate_for_host_safety(
                    123, 123, grace_seconds=1.0
                )
        self.assertIn("exited during grace", action)
        kill.assert_called_once_with(123, signal.SIGTERM)

    def test_does_not_kill_reused_pid_after_grace(self):
        with mock.patch.object(
            WATCHDOG,
            "proc_start_ticks",
            side_effect=[123, 456],
        ):
            with mock.patch.object(WATCHDOG.os, "kill") as kill:
                action = WATCHDOG.terminate_for_host_safety(
                    123, 123, grace_seconds=1.0
                )
        self.assertIn("identity changed during grace", action)
        kill.assert_called_once_with(123, signal.SIGTERM)


class WatchdogFloorConfigTests(unittest.TestCase):
    def test_explicit_watchdog_floor_overrides_engine_floor(self):
        self.assertEqual(
            WATCHDOG.watchdog_live_floor_gib({
                "graph_mem_guard_floor_mb": 4608,
                "host_pressure_watchdog_floor_mb": 1024,
            }),
            1.0,
        )

    def test_watchdog_floor_defaults_to_engine_floor(self):
        self.assertEqual(
            WATCHDOG.watchdog_live_floor_gib({"graph_mem_guard_floor_mb": 4608}),
            4.5,
        )

    def test_nonpositive_watchdog_floor_is_rejected(self):
        with self.assertRaises(ValueError):
            WATCHDOG.watchdog_live_floor_gib({"host_pressure_watchdog_floor_mb": 0})


if __name__ == "__main__":
    unittest.main()
