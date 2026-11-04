#!/usr/bin/env python3
"""Synthetic checks for the guarded inference host-swap watchdog."""

from pathlib import Path
import importlib.util
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "host_pressure_watchdog.py"
SPEC = importlib.util.spec_from_file_location("host_pressure_watchdog", MODULE_PATH)
WATCHDOG = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(WATCHDOG)


class PressureReasonTests(unittest.TestCase):
    def reasons(self, **overrides):
        values = {
            "memavailable_gib": 5.5,
            "vm_swap_gib": 0.0,
            "swap_used_gib": 1.317,
            "baseline_swap_gib": 1.317,
            "psi_full_avg10_pct": 1.0,
            "full_psi_streak": 0,
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

    def test_first_swapout_sample_trips(self):
        reasons = self.reasons(swapout_mib_s=1.1, swapout_streak=1)
        self.assertTrue(any("swap-out" in reason for reason in reasons))

    def test_first_psi_full_sample_trips(self):
        reasons = self.reasons(psi_full_avg10_pct=1.1, full_psi_streak=1)
        self.assertTrue(any("PSI full" in reason for reason in reasons))

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


if __name__ == "__main__":
    unittest.main()
