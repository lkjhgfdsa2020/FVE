#!/usr/bin/env python3
# pv_forecast.py
#
# Daily PV forecast (today + tomorrow) using Open-Meteo hourly GTI (preferred) or shortwave radiation fallback,
# plus a simple PR model (from pr_calendar.json or config.json).
#
# Outputs:
# - forecast_outputs/pv_hourly_forecast_YYYY-MM-DD.csv
# - forecast_plots/pv_forecast_YYYY-MM-DD.png
# - forecasts/forecast_daily_summary.csv  (upsert by Date)
#
# Summary CSV columns:
# Date,PredictionToday,ActualToday,PredictionTomorrow,SwitchOff,SwitchOn
#
# Switch window:
# - monthly OFF budget = 20% of month hours
# - allocate remaining budget across remaining days
# - optionally weight by today's predicted kWh vs current month mean predicted kWh
# - pick best contiguous window (maximizing sum pv_kw_pred) within daylight-ish hours

from __future__ import annotations

import argparse
import calendar
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# Headless plotting for CI / GitHub Actions
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------
# Config
# ---------------------------

@dataclass
class Config:
    latitude: float
    longitude: float
    timezone: str = "Europe/Prague"

    pv_kwp: float = 8.2
    tilt_deg: float = 25.0
    azimuth_deg_from_north: float = 221.0

    # Default PR if pr_calendar is missing
    performance_ratio: float = 0.82

    # Optional per-day cap for switch-off window (hours). If omitted, we allow up to 24.
    # Set this in config.json if you want an absolute cap.
    max_off_hours: float = 24.0

    # Optional daylight window for searching best switch interval (local hours)
    switch_search_start_hour: int = 7
    switch_search_end_hour: int = 19

    # Paths
    pr_calendar_path: str = "pr_calendar.json"
    config_path: str = "config.json"


def load_config(path: str = "config.json") -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing config file: {p.resolve()}")

    j = json.loads(p.read_text(encoding="utf-8"))
    return Config(
        latitude=float(j["latitude"]),
        longitude=float(j["longitude"]),
        timezone=str(j.get("timezone", "Europe/Prague")),
        pv_kwp=float(j.get("pv_kwp", 8.2)),
        tilt_deg=float(j.get("tilt_deg", 25.0)),
        azimuth_deg_from_north=float(j.get("azimuth_deg_from_north", 221.0)),
        performance_ratio=float(j.get("performance_ratio", 0.82)),
        max_off_hours=float(j.get("max_off_hours", 24.0)),
        switch_search_start_hour=int(j.get("switch_search_start_hour", 7)),
        switch_search_end_hour=int(j.get("switch_search_end_hour", 19)),
        pr_calendar_path=str(j.get("pr_calendar_path", "pr_calendar.json")),
        config_path=str(path),
    )


# ---------------------------
# PR calendar loading
# ---------------------------

def _load_pr_calendar(path: str) -> Any | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def pr_for_date(pr_cal: Any | None, cfg: Config, day: date) -> float:
    """
    Tries to support common formats:
    1) {"monthly": {"01": 0.75, "02": 0.78, ...}}
    2) {"monthly": {"1": 0.75, "2": 0.78, ...}}
    3) {"2025-03": 0.82, "2025-04": 0.84, ...}
    4) [{"month":"2025-03","pr":0.82}, ...]
    5) [{"month":3,"pr":0.82}, ...]  (applies to any year)
    Fallback: cfg.performance_ratio
    """
    default = float(cfg.performance_ratio)

    if pr_cal is None:
        return default

    m = day.month
    ym = f"{day.year:04d}-{day.month:02d}"

    try:
        # dict formats
        if isinstance(pr_cal, dict):
            # monthly dict under key
            if "monthly" in pr_cal and isinstance(pr_cal["monthly"], dict):
                md = pr_cal["monthly"]
                for k in (f"{m:02d}", str(m)):
                    if k in md:
                        return float(md[k])

            # direct mapping by YYYY-MM
            if ym in pr_cal:
                return float(pr_cal[ym])

            # direct mapping by month only (rare)
            for k in (f"{m:02d}", str(m)):
                if k in pr_cal:
                    return float(pr_cal[k])

        # list formats
        if isinstance(pr_cal, list):
            # try match YYYY-MM first
            for item in pr_cal:
                if not isinstance(item, dict):
                    continue
                if str(item.get("month", "")).strip() == ym:
                    return float(item.get("pr", default))

            # fallback: match month number
            for item in pr_cal:
                if not isinstance(item, dict):
                    continue
                mm = item.get("month", None)
                if mm is None:
                    continue
                if isinstance(mm, int) and mm == m:
                    return float(item.get("pr", default))
                if isinstance(mm, str) and mm.strip() in (f"{m:02d}", str(m)):
                    return float(item.get("pr", default))

    except Exception:
        return default

    return default


# ---------------------------
# Open-Meteo
# ---------------------------

def fetch_open_meteo_forecast(cfg: Config) -> pd.DataFrame:
    """
    Fetch hourly forecast. We prefer hourly "global_tilted_irradiance" (GTI) if available.
    Some deployments may not return it; we fallback to "shortwave_radiation".
    """
    url = "https://api.open-meteo.com/v1/forecast"

    hourly_vars = [
        # preferred
        "global_tilted_irradiance",
        # fallback
        "shortwave_radiation",
        # optional context
        "cloud_cover",
    ]

    params = {
        "latitude": cfg.latitude,
        "longitude": cfg.longitude,
        "timezone": cfg.timezone,
        "hourly": ",".join(hourly_vars),
        "forecast_days": 2,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()

    if "hourly" not in j or "time" not in j["hourly"]:
        raise RuntimeError("Open-Meteo response missing hourly time series.")

    times = pd.to_datetime(j["hourly"]["time"])
    df = pd.DataFrame({"time": times})
    df["date"] = df["time"].dt.date

    # Some keys may be missing depending on API return
    gti = j["hourly"].get("global_tilted_irradiance", None)
    swr = j["hourly"].get("shortwave_radiation", None)

    if gti is not None:
        df["irr_wm2"] = pd.to_numeric(gti, errors="coerce")
        df["irr_source"] = "gti"
    elif swr is not None:
        df["irr_wm2"] = pd.to_numeric(swr, errors="coerce")
        df["irr_source"] = "shortwave_radiation"
    else:
        raise RuntimeError("Open-Meteo did not return global_tilted_irradiance nor shortwave_radiation.")

    if "cloud_cover" in j["hourly"]:
        df["cloud_cover"] = pd.to_numeric(j["hourly"]["cloud_cover"], errors="coerce")
    else:
        df["cloud_cover"] = pd.NA

    # Keep only today + tomorrow
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ---------------------------
# PV power model (simple)
# ---------------------------

def pv_kw_from_irr(cfg: Config, irr_wm2: float, pr: float) -> float:
    """
    Simple linear model:
    PV(kW) ~ pv_kwp * (irradiance / 1000) * PR
    """
    if irr_wm2 is None:
        return 0.0
    try:
        x = float(irr_wm2)
        if x <= 0:
            return 0.0
        return float(cfg.pv_kwp) * (x / 1000.0) * float(pr)
    except Exception:
        return 0.0


# ---------------------------
# Summary CSV helpers
# ---------------------------

def _round2(x: float) -> float:
    try:
        return float(round(float(x), 2))
    except Exception:
        return float("nan")


def _parse_hhmm(s: Any) -> dtime | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%H:%M").time()
    except Exception:
        return None


def _hours_between_hhmm(off_s: Any, on_s: Any) -> float:
    """Duration in hours within the same day. If invalid -> 0."""
    t1 = _parse_hhmm(off_s)
    t2 = _parse_hhmm(on_s)
    if not t1 or not t2:
        return 0.0
    dt1 = datetime.combine(date(2000, 1, 1), t1)
    dt2 = datetime.combine(date(2000, 1, 1), t2)
    if dt2 <= dt1:
        return 0.0
    return (dt2 - dt1).total_seconds() / 3600.0


def _load_existing_summary(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = df["Date"].astype(str)
        return df
    except Exception:
        return None


def upsert_daily_forecast_summary(
    run_day: date,
    pred_today: float,
    pred_tomorrow: float,
    actual_today: float | None = None,
    switch_off: str = "",
    switch_on: str = "",
    out_csv: str = "forecasts/forecast_daily_summary.csv",
) -> Path:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame(
        [{
            "Date": run_day.isoformat(),
            "PredictionToday": _round2(pred_today),
            "ActualToday": (pd.NA if actual_today is None else _round2(actual_today)),
            "PredictionTomorrow": _round2(pred_tomorrow),
            "SwitchOff": (switch_off or ""),
            "SwitchOn": (switch_on or ""),
        }]
    )

    required = ["Date", "PredictionToday", "ActualToday", "PredictionTomorrow", "SwitchOff", "SwitchOn"]

    if out_path.exists():
        old = pd.read_csv(out_path)

        # Ensure required columns exist
        for c in required:
            if c not in old.columns:
                old[c] = pd.NA

        old["Date"] = old["Date"].astype(str)

        # Remove old row for run_day and append new row
        old = old[old["Date"] != run_day.isoformat()]
        merged = pd.concat([old, new_row], ignore_index=True)
    else:
        merged = new_row

    merged = merged[required].sort_values("Date")

    # Write
    merged.to_csv(out_path, index=False)
    return out_path


# ---------------------------
# Switch window recommendation
# ---------------------------

def recommend_switch_window(
    hourly_df: pd.DataFrame,
    run_day: date,
    pred_today_kwh: float,
    cfg: Config,
    summary_existing: pd.DataFrame | None,
) -> tuple[str, str]:
    """
    Returns (SwitchOff, SwitchOn) in 'HH:MM' local time strings, or ("","") if not recommended.

    Rules:
    - monthly off-budget = 20% of month hours
    - compute already planned off-hours from existing summary CSV
    - allocate remaining budget to today:
        base = remaining_budget / remaining_days
        weight by today's kWh vs month mean predicted kWh (clamped 0.5..2.0)
    - choose best contiguous window length L hours maximizing sum pv_kw_pred within search hours
    """
    year, month = run_day.year, run_day.month
    days_in_month = calendar.monthrange(year, month)[1]
    month_key = f"{year:04d}-{month:02d}"

    monthly_off_budget_hours = 0.20 * days_in_month * 24.0

    used_off_hours = 0.0
    month_mean_pred = None

    if summary_existing is not None and not summary_existing.empty:
        s = summary_existing.copy()
        if "Date" in s.columns:
            s["Date"] = s["Date"].astype(str)
            s_month = s[s["Date"].str.startswith(month_key)]
        else:
            s_month = pd.DataFrame()

        if not s_month.empty:
            if "SwitchOff" in s_month.columns and "SwitchOn" in s_month.columns:
                used_off_hours = float(
                    s_month.apply(
                        lambda r: _hours_between_hhmm(r.get("SwitchOff"), r.get("SwitchOn")), axis=1
                    ).sum()
                )

            if "PredictionToday" in s_month.columns:
                preds = pd.to_numeric(s_month["PredictionToday"], errors="coerce").dropna()
                if len(preds) > 0:
                    month_mean_pred = float(preds.mean())

    remaining_budget = max(0.0, monthly_off_budget_hours - used_off_hours)

    # Remaining days including today
    remaining_days = max(1, (days_in_month - run_day.day + 1))
    base_alloc = remaining_budget / remaining_days  # hours/day

    # Weight by strength of today vs month mean
    if month_mean_pred is None or month_mean_pred <= 0:
        weight = 1.0
    else:
        weight = float(pred_today_kwh) / float(month_mean_pred)
        weight = max(0.5, min(2.0, weight))

    target_hours = base_alloc * weight

    # Respect optional per-day cap
    target_hours = min(float(cfg.max_off_hours), target_hours)

    # If no budget -> no switching
    if target_hours <= 0.25:
        return ("", "")

    # Use integer-hour window for robustness
    L = int(round(target_hours))
    L = max(1, L)

    today = hourly_df[hourly_df["date"] == run_day].copy()
    if today.empty:
        return ("", "")

    today = today.sort_values("time")

    # Search window (local)
    start_h = int(cfg.switch_search_start_hour)
    end_h = int(cfg.switch_search_end_hour)
    today = today[(today["time"].dt.hour >= start_h) & (today["time"].dt.hour <= end_h)].copy()

    if len(today) < L:
        # If not enough points in search window, try full day
        today = hourly_df[hourly_df["date"] == run_day].sort_values("time").copy()
        if len(today) < L:
            return ("", "")

    pv = pd.to_numeric(today["pv_kw_pred"], errors="coerce").fillna(0.0).to_numpy()

    best_sum = -1.0
    best_i = None
    for i in range(0, len(pv) - L + 1):
        s = float(pv[i : i + L].sum())
        if s > best_sum:
            best_sum = s
            best_i = i

    if best_i is None:
        return ("", "")

    start_dt = today.iloc[best_i]["time"].to_pydatetime()
    end_dt = today.iloc[best_i + L - 1]["time"].to_pydatetime() + timedelta(hours=1)

    return (start_dt.strftime("%H:%M"), end_dt.strftime("%H:%M"))


# ---------------------------
# Main forecasting pipeline
# ---------------------------

def build_hourly_forecast(cfg: Config, pr_cal: Any | None, forecast_df: pd.DataFrame) -> pd.DataFrame:
    df = forecast_df.copy()
    # add PR used per row (by date)
    df["pr_used"] = df["date"].apply(lambda d: pr_for_date(pr_cal, cfg, d))

    # pv power prediction
    df["pv_kw_pred"] = [
        pv_kw_from_irr(cfg, irr, pr)
        for irr, pr in zip(df["irr_wm2"].tolist(), df["pr_used"].tolist())
    ]

    return df


def daily_kwh(df_hourly: pd.DataFrame) -> pd.DataFrame:
    # hourly values are kW for the hour → kWh per hour ~ kW * 1h
    g = df_hourly.groupby("date", as_index=False)["pv_kw_pred"].sum()
    g = g.rename(columns={"pv_kw_pred": "pv_kwh_pred"})
    return g


def save_hourly_csv(df_hourly: pd.DataFrame, run_day: date) -> Path:
    out_dir = Path("forecast_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pv_hourly_forecast_{run_day.isoformat()}.csv"

    # Keep a clean export
    export = df_hourly.copy()
    export["time"] = export["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    export["pv_kw_pred"] = export["pv_kw_pred"].astype(float)

    export.to_csv(out_path, index=False)
    return out_path


def save_plot(df_hourly: pd.DataFrame, run_day: date) -> Path:
    out_dir = Path("forecast_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pv_forecast_{run_day.isoformat()}.png"

    # plot today + tomorrow
    df = df_hourly.copy()
    plt.figure(figsize=(12, 5))
    plt.plot(df["time"], df["pv_kw_pred"], label="PV kW pred")
    plt.title(f"PV forecast (hourly) – generated {run_day.isoformat()}")
    plt.xlabel("Time")
    plt.ylabel("kW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Run date YYYY-MM-DD (default: today in local TZ)", default=None)
    ap.add_argument("--out-summary", help="Summary CSV path", default="forecasts/forecast_daily_summary.csv")
    args = ap.parse_args()

    cfg = load_config("config.json")
    pr_cal = _load_pr_calendar(cfg.pr_calendar_path)

    # Determine run day (local) – for Actions this is fine because TZ in workflow is Prague
    if args.date:
        run_day = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        run_day = date.today()

    # 1) Fetch forecast
    raw = fetch_open_meteo_forecast(cfg)

    # 2) Build hourly PV model
    hourly = build_hourly_forecast(cfg, pr_cal, raw)

    # 3) Daily kWh
    daily = daily_kwh(hourly)

    # Only keep today + tomorrow rows
    today = run_day
    tomorrow = run_day + timedelta(days=1)

    day_rows = daily[daily["date"].isin([today, tomorrow])].copy()
    if day_rows.empty or len(day_rows) < 2:
        raise RuntimeError("Did not get both today and tomorrow from forecast daily aggregation.")

    today_kwh = float(day_rows.loc[day_rows["date"] == today, "pv_kwh_pred"].iloc[0])
    tomorrow_kwh = float(day_rows.loc[day_rows["date"] == tomorrow, "pv_kwh_pred"].iloc[0])

    # Print console summary
    print("\nPredikce denní výroby (kWh):")
    print(day_rows.to_string(index=False))

    pr_used_today = pr_for_date(pr_cal, cfg, today)
    pr_used_tomorrow = pr_for_date(pr_cal, cfg, tomorrow)
    pr_used_df = pd.DataFrame([{"date": today, "pr_used": pr_used_today}, {"date": tomorrow, "pr_used": pr_used_tomorrow}])
    print("\nPoužité PR (median za den):")
    print(pr_used_df.to_string(index=False))

    # 4) Save hourly CSV + plot (useful for debugging)
    hourly_csv = save_hourly_csv(hourly, run_day)
    print(f"\nCSV uloženo: {hourly_csv.as_posix()}")

    plot_path = save_plot(hourly, run_day)
    print(f"Graf uložen: {plot_path.as_posix()}")

    # 5) Load existing summary for budget logic
    summary_path = Path(args.out_summary)
    existing = _load_existing_summary(summary_path)

    # 6) Recommend switch window (for TODAY only)
    sw_off, sw_on = recommend_switch_window(
        hourly_df=hourly,
        run_day=today,
        pred_today_kwh=today_kwh,
        cfg=cfg,
        summary_existing=existing,
    )

    # 7) Upsert daily summary (ActualToday stays empty for now)
    out_path = upsert_daily_forecast_summary(
        run_day=today,
        pred_today=today_kwh,
        pred_tomorrow=tomorrow_kwh,
        actual_today=None,
        switch_off=sw_off,
        switch_on=sw_on,
        out_csv=args.out_summary,
    )
    print(f"Souhrn uložen/aktualizován: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
