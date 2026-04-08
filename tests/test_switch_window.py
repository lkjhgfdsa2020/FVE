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
        pv_kw = 0.5
        if 10 <= i <= 14:
            pv_kw = 5.2
        rows.append({"time": t, "date": t.date(), "pv_kw_pred": pv_kw})
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


def _sim_steps_base_then_equal_candidates(*args, **kwargs):
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

    def test_falls_back_to_summary_when_availability_unavailable(self) -> None:
        summary_existing = pd.DataFrame(
            [
                {
                    "Date": "2026-03-01",
                    "PredictionToday": 25.0,
                    "ActualToday": "",
                    "PredictionTomorrow": 20.0,
                    "SwitchOff": "12:00",
                    "SwitchOn": "14:00",
                }
            ]
        )
        with patch.object(pf, "_used_off_hours_from_availability", return_value=None):
            with patch.object(pf, "_simulate_day", side_effect=_sim_base_then_better_off):
                off, on, trace = pf.recommend_switch_window_smart(
                    hourly_df=self.hourly,
                    run_day=self.run_day,
                    pred_today_kwh=30.0,
                    cfg=self.cfg,
                    summary_existing=summary_existing,
                    soc_start_pct=20.0,
                )
        self.assertNotEqual((off, on), ("", ""))
        self.assertEqual(trace.get("decision"), "switch")
        self.assertEqual(trace.get("used_off_hours_source"), "summary_csv")
        self.assertEqual(trace.get("used_off_hours_so_far"), 2.0)

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
        self.assertGreaterEqual(int(off.split(":")[0]), 10)
        self.assertLessEqual(int(on.split(":")[0]), 15)

    def test_negative_spot_prices_shift_window_to_quarter_hour_valley(self) -> None:
        price_rows = []
        start = datetime(self.run_day.year, self.run_day.month, self.run_day.day, 0, 0, 0)
        for i in range(96):
            t = start + timedelta(minutes=15 * i)
            price = 200.0
            if datetime(self.run_day.year, self.run_day.month, self.run_day.day, 11, 15, 0) <= t < datetime(
                self.run_day.year, self.run_day.month, self.run_day.day, 12, 15, 0
            ):
                price = -100.0
            price_rows.append({"time": pd.Timestamp(t), "price_czk": price})
        spot_df = pd.DataFrame(price_rows)

        with patch.object(pf, "_used_off_hours_from_availability", return_value=10.0):
            with patch.object(pf, "fetch_spot_prices_qh", return_value=spot_df):
                with patch.object(pf, "_simulate_steps", side_effect=_sim_steps_base_then_equal_candidates):
                    off, on, trace = pf.recommend_switch_window_smart(
                        hourly_df=self.hourly,
                        run_day=self.run_day,
                        pred_today_kwh=30.0,
                        cfg=self.cfg,
                        summary_existing=None,
                        soc_start_pct=20.0,
                    )

        self.assertEqual(trace.get("decision"), "switch")
        self.assertEqual(trace.get("spot_price_mode"), "quarter_hour")
        self.assertEqual(off, "11:15")
        self.assertEqual(on, "12:15")
        self.assertEqual(trace["best"]["price_min_czk"], -100.0)
        self.assertEqual(trace["best"]["price_negative_points"], 4)


if __name__ == "__main__":
    unittest.main()
