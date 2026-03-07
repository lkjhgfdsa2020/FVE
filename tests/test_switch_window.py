import unittest
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pandas as pd

import pv_forecast as pf


def _hourly_df(run_day: date) -> pd.DataFrame:
    start = datetime(run_day.year, run_day.month, run_day.day, 0, 0, 0)
    rows = []
    for i in range(24):
        t = start + timedelta(hours=i)
        rows.append({"time": t, "date": t.date(), "pv_kw_pred": 3.0})
    return pd.DataFrame(rows)


def _sim_base_then_better_off(*args, **kwargs):
    off_start_idx = args[2]
    if off_start_idx is None:
        return {
            "export_kwh": 0.0,
            "spill_kwh": 1.0,
            "soc_end_frac": 0.95,
            "first_full_hour": None,
        }
    return {
        "export_kwh": 5.0,
        "spill_kwh": 0.0,
        "soc_end_frac": 0.95,
        "first_full_hour": None,
    }


class TestSwitchWindowAvailabilityGate(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = pf.load_config("config.json")
        self.run_day = date(2026, 3, 10)
        self.hourly = _hourly_df(self.run_day)

    def test_returns_empty_when_availability_unavailable(self) -> None:
        with patch.object(pf, "_used_off_hours_from_availability", return_value=None):
            off, on, trace = pf.recommend_switch_window_smart(
                hourly_df=self.hourly,
                run_day=self.run_day,
                pred_today_kwh=30.0,
                cfg=self.cfg,
                summary_existing=None,
                soc_start_pct=20.0,
            )
        self.assertEqual((off, on), ("", ""))
        self.assertEqual(trace.get("reason"), "availability_unavailable")

    def test_returns_empty_when_allowance_exhausted(self) -> None:
        # Day 10 with 10% allowance => 10 * 24 * 0.10 = 24h allowance so far.
        with patch.object(pf, "_used_off_hours_from_availability", return_value=24.0):
            off, on, trace = pf.recommend_switch_window_smart(
                hourly_df=self.hourly,
                run_day=self.run_day,
                pred_today_kwh=30.0,
                cfg=self.cfg,
                summary_existing=None,
                soc_start_pct=20.0,
            )
        self.assertEqual((off, on), ("", ""))
        self.assertEqual(trace.get("reason"), "alloc_below_minimum")

    def test_can_recommend_window_when_allowance_remains(self) -> None:
        with patch.object(pf, "_used_off_hours_from_availability", return_value=10.0):
            with patch.object(pf, "_simulate_day", side_effect=_sim_base_then_better_off):
                off, on, trace = pf.recommend_switch_window_smart(
                    hourly_df=self.hourly,
                    run_day=self.run_day,
                    pred_today_kwh=30.0,
                    cfg=self.cfg,
                    summary_existing=None,
                    soc_start_pct=20.0,
                )
        self.assertNotEqual((off, on), ("", ""))
        self.assertEqual(trace.get("decision"), "switch")


if __name__ == "__main__":
    unittest.main()
