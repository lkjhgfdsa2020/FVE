from datetime import date

import pandas as pd

import pv_forecast as pf
import solax_update_actuals as sua


def test_upsert_daily_forecast_summary_adds_enabled_column_for_new_file(tmp_path) -> None:
    out_csv = tmp_path / "forecast_daily_summary.csv"

    pf.upsert_daily_forecast_summary(
        run_day=date(2026, 4, 4),
        pred_today=28.5,
        pred_tomorrow=31.2,
        switch_off="12:00",
        switch_on="15:00",
        out_csv=str(out_csv),
    )

    df = pd.read_csv(out_csv, keep_default_na=False)
    assert list(df.columns) == pf.SUMMARY_COLUMNS
    assert df.loc[0, "Enabled"] == ""
    assert df.loc[0, "SwitchOff"] == "12:00"
    assert df.loc[0, "SwitchOn"] == "15:00"


def test_upsert_daily_forecast_summary_preserves_existing_enabled_value(tmp_path) -> None:
    out_csv = tmp_path / "forecast_daily_summary.csv"
    out_csv.write_text(
        "\n".join(
            [
                "Date,PredictionToday,ActualToday,PredictionTomorrow,SwitchOff,SwitchOn,Enabled",
                "2026-04-03,20.0,19.5,21.0,10:00,12:00,false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    pf.upsert_daily_forecast_summary(
        run_day=date(2026, 4, 4),
        pred_today=28.5,
        pred_tomorrow=31.2,
        switch_off="12:00",
        switch_on="15:00",
        out_csv=str(out_csv),
    )

    df = pd.read_csv(out_csv, keep_default_na=False)
    row = df.loc[df["Date"] == "2026-04-03"].iloc[0]
    assert str(row["Enabled"]).lower() == "false"


def test_upsert_daily_forecast_summary_preserves_existing_actual_for_same_day(tmp_path) -> None:
    out_csv = tmp_path / "forecast_daily_summary.csv"
    out_csv.write_text(
        "\n".join(
            [
                "Date,PredictionToday,ActualToday,PredictionTomorrow,SwitchOff,SwitchOn,Enabled",
                "2026-05-11,,25.0,,,,false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    pf.upsert_daily_forecast_summary(
        run_day=date(2026, 5, 11),
        pred_today=31.4,
        pred_tomorrow=36.8,
        switch_off="",
        switch_on="",
        out_csv=str(out_csv),
    )

    df = pd.read_csv(out_csv, keep_default_na=False)
    row = df.loc[df["Date"] == "2026-05-11"].iloc[0]
    assert row["PredictionToday"] == 31.4
    assert row["ActualToday"] == 25.0
    assert row["PredictionTomorrow"] == 36.8
    assert str(row["Enabled"]).lower() == "false"


def test_ensure_summary_columns_accepts_legacy_and_new_layouts() -> None:
    legacy = pd.DataFrame(
        [{"Date": "2026-04-04", "PredictionToday": 28.5, "ActualToday": "", "PredictionTomorrow": 31.2, "SwitchOff": "", "SwitchOn": ""}]
    )
    upgraded = sua.ensure_summary_columns(legacy)
    assert list(upgraded.columns) == sua.SUMMARY_COLUMNS
    assert pd.isna(upgraded.loc[0, "Enabled"])

    current = pd.DataFrame(
        [{"Date": "2026-04-04", "PredictionToday": 28.5, "ActualToday": "", "PredictionTomorrow": 31.2, "SwitchOff": "", "SwitchOn": "", "Enabled": "false"}]
    )
    preserved = sua.ensure_summary_columns(current)
    assert str(preserved.loc[0, "Enabled"]).lower() == "false"
