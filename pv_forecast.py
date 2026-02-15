#!/usr/bin/env python3
# pv_forecast.py
#
# Daily PV forecast (today + tomorrow) using Open-Meteo hourly irradiance (GTI preferred, SWR fallback),
# PR model (from pr_calendar.json or config.json), and writes:
#
# - forecast_outputs/pv_hourly_forecast_YYYY-MM-DD.csv
# - forecast_plots/pv_forecast_YYYY-MM-DD.png
# - forecasts/forecast_daily_summary.csv  (upsert by Date)
#
# Summary CSV columns:
# Date,PredictionToday,ActualToday,PredictionTomorrow,SwitchOff,SwitchOn
#
# Switch window recommendation (SMART):
# - Models your controller behavior in Backup mode:
#   ON  (controller connected): House -> Charge battery -> Export ONLY when battery is 100%
#   OFF (controller disconnected): House -> Export (up to limit) -> Charge battery from remaining
#
# - Chooses the best OFF window by hourly simulation & scoring:
#   maximize export, minimize spill/curtailment, keep evening SoC target (default 90%),
#   and obey monthly OFF budget (plant ON >= 80% of month).
#
# - SoC is fetched from SolaXCloud API (realtimeInfo/get) if env vars exist:
#   SOLAX_BASE_URL, SOLAX_TOKEN_ID, SOLAX_WIFI_SN
#   If SoC is null/unavailable -> default 20%.
#
# SwitchOff/SwitchOn are empty strings if we recommend not switching off at all.

from __future__ import annotations

import argparse
import calendar
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any, Optional

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

    # Export + house baseline
    export_limit_kw: float = 3.65
    baseload_kw: float = 0.50

    # Battery model
    battery_kwh_total: float = 11.6
    battery_usable_frac: float = 0.90  # usable = total * frac
    target_soc_evening: float = 0.85   # target at end of day (usable fraction)
    soc_default_pct: float = 20.0      # used if API returns null/unavailable

    # Switch window constraints
    max_off_hours: float = 10.0
    switch_search_start_hour: int = 8
    switch_search_end_hour: int = 16

    # Only consider switching on strong days
    min_pred_kwh_for_switch: float = 20.0

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
        export_limit_kw=float(j.get("export_limit_kw", 3.65)),
        baseload_kw=float(j.get("baseload_kw", 0.50)),
        battery_kwh_total=float(j.get("battery_kwh_total", 11.6)),
        battery_usable_frac=float(j.get("battery_usable_frac", 0.90)),
        target_soc_evening=float(j.get("target_soc_evening", 0.85)),
        soc_default_pct=float(j.get("soc_default_pct", 20.0)),
        max_off_hours=float(j.get("max_off_hours", 10.0)),
        switch_search_start_hour=int(j.get("switch_search_start_hour", 8)),
        switch_search_end_hour=int(j.get("switch_search_end_hour", 16)),
        min_pred_kwh_for_switch=float(j.get("min_pred_kwh_for_switch", 20.0)),
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
    Supports common formats:
      1) {"monthly": {"01": 0.75, "02": 0.78, ...}}
      2) {"2025-03": 0.82, "2025-04": 0.84, ...}
      3) [{"month":"2025-03","pr":0.82}, ...]
      4) [{"month":3,"pr":0.82}, ...]  (applies to any year)
    Fallback: cfg.performance_ratio
    """
    default = float(cfg.performance_ratio)
    if pr_cal is None:
        return default

    m = day.month
    ym = f"{day.year:04d}-{day.month:02d}"

    try:
        if isinstance(pr_cal, dict):
            if "monthly" in pr_cal and isinstance(pr_cal["monthly"], dict):
                md = pr_cal["monthly"]
                for k in (f"{m:02d}", str(m)):
                    if k in md:
                        return float(md[k])

            if ym in pr_cal:
                return float(pr_cal[ym])

            for k in (f"{m:02d}", str(m)):
                if k in pr_cal:
                    return float(pr_cal[k])

        if isinstance(pr_cal, list):
            for item in pr_cal:
                if isinstance(item, dict) and str(item.get("month", "")).strip() == ym:
                    return float(item.get("pr", default))

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
# SolaXCloud SoC
# ---------------------------

def fetch_solax_soc_percent(cfg: Config) -> float:
    """
    Fetch SoC (%) from SolaXCloud realtimeInfo/get (env vars).
    If missing / null / error => fallback cfg.soc_default_pct.
    """
    import os

    base_url = os.getenv("SOLAX_BASE_URL", "").strip()
    token_id = os.getenv("SOLAX_TOKEN_ID", "").strip()
    wifi_sn = os.getenv("SOLAX_WIFI_SN", "").strip()

    fallback = float(cfg.soc_default_pct)

    if not (base_url and token_id and wifi_sn):
        return fallback

    try:
        url = f"{base_url.rstrip('/')}/api/v2/dataAccess/realtimeInfo/get"
        headers = {"tokenId": token_id, "Content-Type": "application/json"}
        payload = {"wifiSn": wifi_sn}

        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        j = r.json()

        if not j.get("success", False):
            return fallback

        result = j.get("result", {}) or {}
        soc = result.get("soc", None)
        if soc is None:
            return fallback

        v = float(soc)
        if v < 0:
            return fallback
        return min(100.0, v)
    except Exception:
        return fallback


# ---------------------------
# Open-Meteo
# ---------------------------

def fetch_open_meteo_forecast(cfg: Config) -> pd.DataFrame:
    """
    Fetch hourly forecast. Prefer hourly "global_tilted_irradiance" if available,
    else fallback to "shortwave_radiation".
    """
    url = "https://api.open-meteo.com/v1/forecast"

    hourly_vars = [
        "global_tilted_irradiance",
        "shortwave_radiation",
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

    return df.sort_values("time").reset_index(drop=True)


# ---------------------------
# PV power model (simple)
# ---------------------------

def pv_kw_from_irr(cfg: Config, irr_wm2: float, pr: float) -> float:
    """PV(kW) ~ pv_kwp * (irradiance / 1000) * PR"""
    try:
        x = float(irr_wm2)
        # Robust against NaN/inf or missing values
        if not math.isfinite(x) or x <= 0:
            return 0.0
        return float(cfg.pv_kwp) * (x / 1000.0) * float(pr)
    except Exception:
        return 0.0


# ---------------------------
# Switch window recommendation (SMART simulation)
# ---------------------------

def _month_off_budget_hours(year: int, month: int) -> float:
    days_in_month = calendar.monthrange(year, month)[1]
    return 0.20 * days_in_month * 24.0  # OFF <= 20% => ON >= 80%


def _compute_used_off_hours(summary_existing: pd.DataFrame | None, month_key: str) -> float:
    if summary_existing is None or summary_existing.empty:
        return 0.0
    s = summary_existing.copy()
    if "Date" not in s.columns:
        return 0.0
    s["Date"] = s["Date"].astype(str)
    s_month = s[s["Date"].str.startswith(month_key)].copy()
    if s_month.empty:
        return 0.0
    if "SwitchOff" not in s_month.columns or "SwitchOn" not in s_month.columns:
        return 0.0
    return float(
        s_month.apply(lambda r: _hours_between_hhmm(r.get("SwitchOff"), r.get("SwitchOn")), axis=1).sum()
    )


def _simulate_day(
    pv_kw: list[float],
    hours: list[int],
    off_start_idx: int | None,
    off_len: int,
    cfg: Config,
    soc_start_pct: float,
) -> dict[str, Any]:
    cap = float(cfg.battery_kwh_total) * float(cfg.battery_usable_frac)  # usable kWh
    cap = max(0.1, cap)

    soc0 = max(0.0, min(100.0, float(soc_start_pct)))
    E = (soc0 / 100.0) * cap  # usable energy (kWh)

    export_kwh = 0.0
    spill_kwh = 0.0
    first_full_hour: Optional[int] = None

    off_end_idx = None
    if off_start_idx is not None and off_len > 0:
        off_end_idx = off_start_idx + off_len

    for i, pv in enumerate(pv_kw):
        h = hours[i]
        pv = max(0.0, float(pv))

        after_load = max(0.0, pv - float(cfg.baseload_kw))
        in_off = False
        if off_start_idx is not None and off_end_idx is not None:
            in_off = (i >= off_start_idx) and (i < off_end_idx)

        batt_rem = max(0.0, cap - E)

        if in_off:
            export = min(float(cfg.export_limit_kw), after_load)
            rem = max(0.0, after_load - export)
            charge = min(rem, batt_rem)
            spill = max(0.0, rem - charge)
        else:
            charge = min(after_load, batt_rem)
            rem = max(0.0, after_load - charge)
            if batt_rem <= 1e-9:
                export = min(float(cfg.export_limit_kw), after_load)
                spill = max(0.0, after_load - export)
            else:
                export = 0.0
                spill = rem

        E = min(cap, E + charge)
        export_kwh += export
        spill_kwh += spill

        if first_full_hour is None and E >= cap - 1e-6:
            first_full_hour = int(h)

    return {
        "export_kwh": export_kwh,
        "spill_kwh": spill_kwh,
        "soc_end_frac": E / cap,
        "first_full_hour": first_full_hour,
    }


def recommend_switch_window_smart(
    hourly_df: pd.DataFrame,
    run_day: date,
    pred_today_kwh: float,
    cfg: Config,
    summary_existing: pd.DataFrame | None,
    soc_start_pct: float,
) -> tuple[str, str]:
    if float(pred_today_kwh) < float(cfg.min_pred_kwh_for_switch):
        return ("", "")

    today = hourly_df[hourly_df["date"] == run_day].copy()
    if today.empty:
        return ("", "")

    today = today.sort_values("time")
    pv = pd.to_numeric(today["pv_kw_pred"], errors="coerce").fillna(0.0).tolist()
    hours = today["time"].dt.hour.astype(int).tolist()
    if len(pv) < 8:
        return ("", "")

    year, month = run_day.year, run_day.month
    month_key = f"{year:04d}-{month:02d}"
    budget = _month_off_budget_hours(year, month)
    used = _compute_used_off_hours(summary_existing, month_key)
    remaining_budget = max(0.0, budget - used)

    days_in_month = calendar.monthrange(year, month)[1]
    remaining_days = max(1, days_in_month - run_day.day + 1)

    month_mean_pred = None
    if (
        summary_existing is not None
        and not summary_existing.empty
        and "Date" in summary_existing.columns
        and "PredictionToday" in summary_existing.columns
    ):
        s = summary_existing.copy()
        s["Date"] = s["Date"].astype(str)
        s_month = s[s["Date"].str.startswith(month_key)].copy()
        preds = pd.to_numeric(s_month.get("PredictionToday", pd.Series([], dtype=float)), errors="coerce").dropna()
        if len(preds) > 0:
            month_mean_pred = float(preds.mean())

    if month_mean_pred and month_mean_pred > 0:
        weight = float(pred_today_kwh) / float(month_mean_pred)
        weight = max(0.6, min(2.2, weight))
    else:
        weight = 1.0

    base_alloc = remaining_budget / remaining_days
    alloc_today = min(float(cfg.max_off_hours), base_alloc * weight, remaining_budget)
    if alloc_today < 0.75:
        return ("", "")

    maxL = max(1, int(round(alloc_today)))

    start_h = int(cfg.switch_search_start_hour)
    end_h = int(cfg.switch_search_end_hour)
    candidates: list[int] = [i for i, h in enumerate(hours) if start_h <= h <= end_h]
    if not candidates:
        candidates = list(range(len(pv)))

    base = _simulate_day(pv, hours, None, 0, cfg, soc_start_pct)

    def score(sim: dict[str, Any]) -> float:
        export = float(sim["export_kwh"])
        spill = float(sim["spill_kwh"])
        end_frac = float(sim["soc_end_frac"])
        full_hour = sim.get("first_full_hour", None)

        target = float(cfg.target_soc_evening)
        deficit = max(0.0, target - end_frac)
        penalty_end = 1500.0 * (deficit ** 2)

        penalty_early_full = 0.0
        if full_hour is not None and int(full_hour) < 15:
            penalty_early_full = 2.5 * float(15 - int(full_hour))

        return (3.0 * export) - (4.0 * spill) - penalty_end - penalty_early_full

    base_score = score(base)
    best = {"score": base_score, "off": "", "on": "", "L": 0}

    for L in range(1, maxL + 1):
        for s_idx in candidates:
            e_idx = s_idx + L
            if e_idx > len(pv):
                continue
            sim = _simulate_day(pv, hours, s_idx, L, cfg, soc_start_pct)
            sc = score(sim)
            if sc > best["score"]:
                off_time = today.iloc[s_idx]["time"].to_pydatetime().strftime("%H:%M")
                on_time = (today.iloc[e_idx - 1]["time"].to_pydatetime() + timedelta(hours=1)).strftime("%H:%M")
                best = {"score": sc, "off": off_time, "on": on_time, "L": L}

    if best["L"] == 0 or (best["score"] - base_score) < 1.0:
        return ("", "")

    return (str(best["off"]), str(best["on"]))


# ---------------------------
# Forecast pipeline
# ---------------------------

def build_hourly_forecast(cfg: Config, pr_cal: Any | None, forecast_df: pd.DataFrame) -> pd.DataFrame:
    df = forecast_df.copy()
    df["pr_used"] = df["date"].apply(lambda d: pr_for_date(pr_cal, cfg, d))
    df["pv_kw_pred"] = [
        pv_kw_from_irr(cfg, irr, pr)
        for irr, pr in zip(df["irr_wm2"].tolist(), df["pr_used"].tolist())
    ]
    return df


def daily_kwh(df_hourly: pd.DataFrame) -> pd.DataFrame:
    g = df_hourly.groupby("date", as_index=False)["pv_kw_pred"].sum()
    return g.rename(columns={"pv_kw_pred": "pv_kwh_pred"})


def save_hourly_csv(df_hourly: pd.DataFrame, run_day: date) -> Path:
    out_dir = Path("forecast_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pv_hourly_forecast_{run_day.isoformat()}.csv"

    export = df_hourly.copy()
    export["time"] = export["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    export.to_csv(out_path, index=False)
    return out_path

def build_intraday_30min(cfg: Config, pr_cal: Any | None, hourly_forecast: pd.DataFrame, run_day: date) -> pd.DataFrame:
    """
    Build 30-minute intraday forecast for a specific day.

    Strategy (robust across Pandas versions & tz handling):
      - Slice today's hourly rows.
      - Force timestamps to timezone-naive local time (drop tzinfo if present).
      - Create a 30-min grid 00:00..23:30.
      - Interpolate numeric series using time interpolation on a unified DatetimeIndex.
      - Prefer interpolating pv_kw_pred directly (it must exist if daily kWh is non-zero),
        and additionally interpolate irr_wm2 / cloud_cover when available.
      - step_kwh = pv_kw_pred * 0.5 (kWh per 30-min step)
    """
    cols = ["time", "pv_kw_pred", "step_kwh", "irr_wm2", "cloud_cover", "pr_used", "irr_source"]
    day_df = hourly_forecast[hourly_forecast["date"] == run_day].copy()
    if day_df.empty:
        return pd.DataFrame(columns=cols)

    # Ensure datetime and drop tz-info (avoid tz-aware vs tz-naive union issues)
    day_df["time"] = pd.to_datetime(day_df["time"], errors="coerce")
    try:
        if getattr(day_df["time"].dt, "tz", None) is not None:
            # Convert to local tz (cfg.timezone) if possible, then drop tz
            day_df["time"] = day_df["time"].dt.tz_convert(cfg.timezone).dt.tz_localize(None)
        else:
            # If tz-naive already, keep as-is
            pass
    except Exception:
        # Fallback: just drop tz if it's there
        try:
            day_df["time"] = day_df["time"].dt.tz_localize(None)
        except Exception:
            pass

    day_df = day_df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    start = pd.Timestamp(datetime.combine(run_day, dtime(0, 0)))
    end = pd.Timestamp(datetime.combine(run_day, dtime(23, 30)))
    grid = pd.date_range(start=start, end=end, freq="30min")

    src = day_df.set_index("time")

    # Prepare series to interpolate (force numeric)
    want = []
    for c in ["pv_kw_pred", "irr_wm2", "cloud_cover"]:
        if c in src.columns:
            src[c] = pd.to_numeric(src[c], errors="coerce")
            want.append(c)

    # If pv_kw_pred is missing (shouldn't happen), compute it from irr_wm2 if possible
    if "pv_kw_pred" not in want:
        if "irr_wm2" in src.columns:
            pr = pr_for_date(pr_cal, cfg, run_day)
            src["pv_kw_pred"] = [pv_kw_from_irr(cfg, irr, pr) for irr in src["irr_wm2"].tolist()]
            want.append("pv_kw_pred")
        else:
            # Nothing to do
            out = pd.DataFrame({"time": grid})
            out["pv_kw_pred"] = 0.0
            out["step_kwh"] = 0.0
            out["irr_wm2"] = 0.0
            out["cloud_cover"] = pd.NA
            out["pr_used"] = float(pr_for_date(pr_cal, cfg, run_day))
            out["irr_source"] = "unknown"
            return out[cols]

    # Time interpolation on unified index
    idx = src.index.union(grid)
    tmp = src[want].reindex(idx).sort_index()
    tmp = tmp.interpolate(method="time")
    tmp = tmp.reindex(grid)

    out = pd.DataFrame({"time": grid})
    pr = float(pr_for_date(pr_cal, cfg, run_day))
    out["pr_used"] = pr

    # irr_source constant
    if "irr_source" in day_df.columns and len(day_df["irr_source"].dropna()) > 0:
        out["irr_source"] = str(day_df["irr_source"].dropna().iloc[0])
    else:
        out["irr_source"] = "unknown"

    # pv_kw_pred (prefer interpolated pv)
    out["pv_kw_pred"] = pd.to_numeric(tmp.get("pv_kw_pred", pd.Series(index=grid, data=0.0)), errors="coerce").fillna(0.0)
    out.loc[out["pv_kw_pred"] < 0, "pv_kw_pred"] = 0.0

    # irr_wm2 (if missing, approximate from pv_kw_pred)
    if "irr_wm2" in tmp.columns:
        out["irr_wm2"] = pd.to_numeric(tmp["irr_wm2"], errors="coerce").fillna(0.0)
        out.loc[out["irr_wm2"] < 0, "irr_wm2"] = 0.0
    else:
        # Invert PV model: pv = pv_kwp * (irr/1000) * pr  => irr = pv/(pv_kwp*pr)*1000
        denom = max(1e-9, float(cfg.pv_kwp) * pr)
        out["irr_wm2"] = (out["pv_kw_pred"] / denom) * 1000.0

    if "cloud_cover" in tmp.columns:
        out["cloud_cover"] = pd.to_numeric(tmp["cloud_cover"], errors="coerce")
    else:
        out["cloud_cover"] = pd.NA

    out["step_kwh"] = out["pv_kw_pred"] * 0.5
    return out[cols]


def save_intraday_csv(df_intraday: pd.DataFrame, run_day: date) -> Path:
    out_dir = Path("forecasts") / "intraday"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"forecast_intraday_{run_day.isoformat()}.csv"

    export = df_intraday.copy()
    export["time"] = export["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cols = ["time", "pv_kw_pred", "step_kwh", "irr_wm2", "cloud_cover", "pr_used", "irr_source"]
    # Keep only existing columns in desired order
    cols = [c for c in cols if c in export.columns]
    export = export[cols]
    export.to_csv(out_path, index=False)
    return out_path


def cleanup_intraday_files(keep_days: int = 30) -> int:
    """
    Delete intraday forecast CSVs older than keep_days (based on filename date).
    Returns count of deleted files.
    """
    intraday_dir = Path("forecasts") / "intraday"
    if not intraday_dir.exists():
        return 0

    today = date.today()
    deleted = 0

    for p in intraday_dir.glob("forecast_intraday_*.csv"):
        m = re.search(r"forecast_intraday_(\d{4}-\d{2}-\d{2})\.csv$", p.name)
        if not m:
            continue
        try:
            d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except Exception:
            continue
        if (today - d).days > int(keep_days):
            try:
                p.unlink()
                deleted += 1
            except Exception:
                pass
    return deleted


def save_plot(df_hourly: pd.DataFrame, run_day: date) -> Path:
    out_dir = Path("forecast_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pv_forecast_{run_day.isoformat()}.png"

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
    ap.add_argument("--date", help="Run date YYYY-MM-DD (default: today)", default=None)
    ap.add_argument("--out-summary", help="Summary CSV path", default="forecasts/forecast_daily_summary.csv")
    args = ap.parse_args()

    cfg = load_config("config.json")
    pr_cal = _load_pr_calendar(cfg.pr_calendar_path)

    run_day = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    today = run_day
    tomorrow = run_day + timedelta(days=1)

    soc_pct = fetch_solax_soc_percent(cfg)
    print(f"SoC (start, used): {soc_pct:.1f}% (fallback={cfg.soc_default_pct:.1f}%)")
    print(f"Evening target SoC: {cfg.target_soc_evening * 100:.0f}% (usable)")

    raw = fetch_open_meteo_forecast(cfg)
    hourly = build_hourly_forecast(cfg, pr_cal, raw)
    daily = daily_kwh(hourly)

    day_rows = daily[daily["date"].isin([today, tomorrow])].copy()
    if day_rows.empty or len(day_rows) < 2:
        raise RuntimeError("Did not get both today and tomorrow from forecast aggregation.")

    today_kwh = float(day_rows.loc[day_rows["date"] == today, "pv_kwh_pred"].iloc[0])
    tomorrow_kwh = float(day_rows.loc[day_rows["date"] == tomorrow, "pv_kwh_pred"].iloc[0])

    print("\nPredikce denní výroby (kWh):")
    print(day_rows.to_string(index=False))

    hourly_csv = save_hourly_csv(hourly, run_day)
    print(f"\nCSV uloženo: {hourly_csv.as_posix()}")

    plot_path = save_plot(hourly, run_day)
    print(f"Graf uložen: {plot_path.as_posix()}")

    intraday = build_intraday_30min(cfg, pr_cal, hourly, today)
    intraday_path = save_intraday_csv(intraday, today)
    print(f"Intraday CSV uloženo: {intraday_path.as_posix()}")

    deleted = cleanup_intraday_files(keep_days=30)
    if deleted:
        print(f"Intraday cleanup: smazáno {deleted} souborů (starší než 30 dní)")

    summary_path = Path(args.out_summary)
    existing = _load_existing_summary(summary_path)

    sw_off, sw_on = recommend_switch_window_smart(
        hourly_df=hourly,
        run_day=today,
        pred_today_kwh=today_kwh,
        cfg=cfg,
        summary_existing=existing,
        soc_start_pct=soc_pct,
    )

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


# --- helpers used above but defined after to keep sections readable ---

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
    t1 = _parse_hhmm(off_s)
    t2 = _parse_hhmm(on_s)
    if not t1 or not t2:
        return 0.0
    dt1 = datetime.combine(date(2000, 1, 1), t1)
    dt2 = datetime.combine(date(2000, 1, 1), t2)
    if dt2 <= dt1:
        return 0.0
    return (dt2 - dt1).total_seconds() / 3600.0


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

    required = ["Date", "PredictionToday", "ActualToday", "PredictionTomorrow", "SwitchOff", "SwitchOn"]

    new_row = pd.DataFrame([{
        "Date": run_day.isoformat(),
        "PredictionToday": _round2(pred_today),
        "ActualToday": (pd.NA if actual_today is None else _round2(actual_today)),
        "PredictionTomorrow": _round2(pred_tomorrow),
        "SwitchOff": (switch_off or ""),
        "SwitchOn": (switch_on or ""),
    }])

    if out_path.exists():
        old = pd.read_csv(out_path)
        for c in required:
            if c not in old.columns:
                old[c] = pd.NA
        old["Date"] = old["Date"].astype(str)
        old = old[old["Date"] != run_day.isoformat()]
        merged = pd.concat([old, new_row], ignore_index=True)
    else:
        merged = new_row

    merged = merged[required].sort_values("Date")
    merged.to_csv(out_path, index=False)
    return out_path


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


if __name__ == "__main__":
    main()
