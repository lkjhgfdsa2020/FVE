\
#!/usr/bin/env python3
# pv_forecast.py
#
# Daily PV forecast (today + tomorrow) using Open-Meteo hourly irradiance (GTI preferred, SWR fallback),
# PR model (from pr_calendar.json or config.json).
#
# Outputs:
# - forecast_outputs/pv_hourly_forecast_YYYY-MM-DD.csv              (hourly for 2 days)
# - forecast_plots/pv_forecast_YYYY-MM-DD.png                       (hourly plot for 2 days)
# - forecasts/forecast_daily_summary.csv                             (upsert by Date)
# - forecasts/intraday/forecast_intraday_YYYY-MM-DD.csv              (hourly for TODAY only, for UI)
#
# Summary CSV columns:
# Date,PredictionToday,ActualToday,PredictionTomorrow,SwitchOff,SwitchOn,Enabled
#
# Snow correction (optional):
# - Fetches additional Open-Meteo hourly vars: temperature_2m, rain, snowfall, snow_depth
# - Applies multiplicative snow_factor per hour using snow_runtime.SnowModel
# - Exports snow_factor + snow meteo columns into hourly/intraday CSV for debugging
#
# Notes:
# - Enabled is optional for compatibility with older firmware / CSVs.
# - SwitchOff/SwitchOn are empty if recommendation is "do not switch off".
# - SoC start is fetched from SolaXCloud API if env vars exist; otherwise defaults to soc_default_pct.

from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd
import requests

# Headless plotting for CI / GitHub Actions
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Snow model (optional)
try:
    from snow_runtime import SnowParams, SnowModel
except Exception:  # pragma: no cover
    SnowParams = None  # type: ignore
    SnowModel = None   # type: ignore


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

    performance_ratio: float = 0.82

    export_limit_kw: float = 3.65
    baseload_kw: float = 0.50

    battery_kwh_total: float = 11.6
    battery_usable_frac: float = 0.90

    # evening target SoC (usable fraction)
    target_soc_evening: float = 0.90
    soc_default_pct: float = 20.0

    monthly_off_allowance_frac: float = 0.10
    max_off_hours: float = 10.0
    switch_search_start_hour: int = 8
    switch_search_end_hour: int = 16
    min_pred_kwh_for_switch: float = 20.0
    min_peak_kw_for_switch: float = 4.0
    preferred_switch_start_hour: int = 10
    preferred_max_off_hours: float = 4.0
    spot_price_enabled: bool = True
    spot_price_api_url: str = "https://spotovaelektrina.cz/api/v1/price/get-prices-json-qh"
    spot_price_negative_threshold_czk: float = 0.0
    spot_price_pv_score_tolerance: float = 2.0

    pr_calendar_path: str = "pr_calendar.json"
    config_path: str = "config.json"

    # Snow correction
    snow_enabled: bool = True
    snow_params_path: str = "snow_params.json"


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
        target_soc_evening=float(j.get("target_soc_evening", 0.90)),
        soc_default_pct=float(j.get("soc_default_pct", 20.0)),
        monthly_off_allowance_frac=float(j.get("monthly_off_allowance_frac", 0.10)),
        max_off_hours=float(j.get("max_off_hours", 10.0)),
        switch_search_start_hour=int(j.get("switch_search_start_hour", 8)),
        switch_search_end_hour=int(j.get("switch_search_end_hour", 16)),
        min_pred_kwh_for_switch=float(j.get("min_pred_kwh_for_switch", 20.0)),
        min_peak_kw_for_switch=float(j.get("min_peak_kw_for_switch", 4.0)),
        preferred_switch_start_hour=int(j.get("preferred_switch_start_hour", 10)),
        preferred_max_off_hours=float(j.get("preferred_max_off_hours", 4.0)),
        spot_price_enabled=bool(j.get("spot_price_enabled", True)),
        spot_price_api_url=str(
            j.get("spot_price_api_url", "https://spotovaelektrina.cz/api/v1/price/get-prices-json-qh")
        ),
        spot_price_negative_threshold_czk=float(j.get("spot_price_negative_threshold_czk", 0.0)),
        spot_price_pv_score_tolerance=float(j.get("spot_price_pv_score_tolerance", 2.0)),
        pr_calendar_path=str(j.get("pr_calendar_path", "pr_calendar.json")),
        config_path=str(path),
        snow_enabled=bool(j.get("snow_enabled", True)),
        snow_params_path=str(j.get("snow_params_path", "snow_params.json")),
    )


# ---------------------------
# PR calendar
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
    Fetch hourly forecast for 2 days.
    Always includes: global_tilted_irradiance (preferred) or shortwave_radiation (fallback), cloud_cover,
    plus snow-related fields for snow correction.
    """
    url = "https://api.open-meteo.com/v1/forecast"

    hourly_vars = [
        "global_tilted_irradiance",
        "shortwave_radiation",
        "cloud_cover",
        # snow correction inputs:
        "temperature_2m",
        "rain",
        "snowfall",
        "snow_depth",
    ]

    # NOTE: Open-Meteo's `global_tilted_irradiance` depends on `tilt` + `azimuth` query parameters.
    # - cfg.azimuth_deg_from_north is degrees clockwise from North (0=N, 90=E, 180=S, 270=W).
    # - Open-Meteo expects azimuth as degrees from South (0=S, -90=E, +90=W, ±180=N).
    az_open = cfg.azimuth_deg_from_north - 180.0
    # wrap to [-180, 180]
    az_open = ((az_open + 180.0) % 360.0) - 180.0

    params = {
        "latitude": cfg.latitude,
        "longitude": cfg.longitude,
        "timezone": cfg.timezone,
        "hourly": ",".join(hourly_vars),
        "daily": "sunrise,sunset",
        "forecast_days": 2,
        "tilt": cfg.tilt_deg,
        "azimuth": az_open,
    }

    last_err: Exception | None = None
    last_status: int | None = None
    last_body: str = ""
    retry_delays_s = (0.0, 2.0, 5.0)

    for attempt_idx, delay_s in enumerate(retry_delays_s, start=1):
        if delay_s > 0:
            time.sleep(delay_s)
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            j = r.json()
            break
        except requests.HTTPError as exc:
            last_err = exc
            resp = exc.response
            last_status = resp.status_code if resp is not None else None
            last_body = (resp.text or "").strip()[:200] if resp is not None else ""
            should_retry = last_status in {429, 500, 502, 503, 504} and attempt_idx < len(retry_delays_s)
            status_txt = f"HTTP {last_status}" if last_status is not None else "HTTP error"
            print(f"Open-Meteo request failed ({status_txt}, attempt {attempt_idx}/{len(retry_delays_s)})")
            if not should_retry:
                break
        except requests.RequestException as exc:
            last_err = exc
            last_status = None
            last_body = ""
            should_retry = attempt_idx < len(retry_delays_s)
            print(
                f"Open-Meteo request failed ({exc.__class__.__name__}, "
                f"attempt {attempt_idx}/{len(retry_delays_s)})"
            )
            if not should_retry:
                break
    else:  # pragma: no cover
        raise AssertionError("retry loop must either break or return")

    if last_err is not None and "j" not in locals():
        details = []
        if last_status is not None:
            details.append(f"status={last_status}")
        if last_body:
            details.append(f"body={last_body!r}")
        detail_suffix = f" ({', '.join(details)})" if details else ""
        raise RuntimeError(
            "Open-Meteo forecast request failed after retries. "
            f"This currently looks like an upstream outage rather than an API schema change{detail_suffix}."
        ) from last_err

    if "hourly" not in j or "time" not in j["hourly"]:
        raise RuntimeError("Open-Meteo response missing hourly time series.")

    times = pd.to_datetime(j["hourly"]["time"])
    df = pd.DataFrame({"time": times})
    df["date"] = df["time"].dt.date

    daily = j.get("daily", {}) or {}
    daily_dates = pd.to_datetime(pd.Series(daily.get("time", [])), errors="coerce").dt.date
    daily_sunrise = pd.to_datetime(pd.Series(daily.get("sunrise", [])), errors="coerce")
    daily_sunset = pd.to_datetime(pd.Series(daily.get("sunset", [])), errors="coerce")
    if not daily_dates.empty and len(daily_dates) == len(daily_sunrise) == len(daily_sunset):
        sun_map = pd.DataFrame(
            {"date": daily_dates, "sunrise": daily_sunrise, "sunset": daily_sunset}
        ).dropna(subset=["date"])
        if not sun_map.empty:
            sun_map = sun_map.drop_duplicates(subset=["date"], keep="last").set_index("date")
            df["sunrise"] = df["date"].map(sun_map["sunrise"])
            df["sunset"] = df["date"].map(sun_map["sunset"])

    gti = j["hourly"].get("global_tilted_irradiance", None)
    swr = j["hourly"].get("shortwave_radiation", None)

    if gti is not None:
        df["irr_wm2"] = pd.Series(pd.to_numeric(gti, errors="coerce")).fillna(0.0)
        df["irr_source"] = "gti"
    elif swr is not None:
        df["irr_wm2"] = pd.Series(pd.to_numeric(swr, errors="coerce")).fillna(0.0)
        df["irr_source"] = "shortwave_radiation"
    else:
        raise RuntimeError("Open-Meteo did not return GTI nor shortwave_radiation.")

    def _col(name: str) -> pd.Series:
        v = j["hourly"].get(name, None)
        if v is None:
            return pd.Series([pd.NA] * len(df))
        return pd.to_numeric(v, errors="coerce")

    df["cloud_cover"] = _col("cloud_cover")
    df["temperature_2m"] = _col("temperature_2m")
    df["rain"] = _col("rain")
    df["snowfall"] = _col("snowfall")
    df["snow_depth"] = _col("snow_depth")

    return df.sort_values("time").reset_index(drop=True)


# ---------------------------
# PV model
# ---------------------------

def pv_kw_from_irr(cfg: Config, irr_wm2: float, pr: float) -> float:
    """PV(kW) ~ pv_kwp * (irradiance/1000) * PR"""
    try:
        x = float(irr_wm2)
        if x <= 0:
            return 0.0
        return float(cfg.pv_kwp) * (x / 1000.0) * float(pr)
    except Exception:
        return 0.0


def apply_snow_correction(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["snow_factor"] = 1.0

    if not cfg.snow_enabled:
        return df
    if SnowParams is None or SnowModel is None:
        return df

    p = Path(cfg.snow_params_path)
    if not p.exists():
        return df

    try:
        sp = SnowParams.from_json(p)
        sm = SnowModel(sp, snow_index0=0.0)

        df = df.sort_values("time").reset_index(drop=True)

        factors = []
        prev_depth = None
        for _, r in df.iterrows():
            temp = r.get("temperature_2m", pd.NA)
            rain = r.get("rain", pd.NA)
            snowfall = r.get("snowfall", pd.NA)
            depth = r.get("snow_depth", pd.NA)
            irr = r.get("irr_wm2", pd.NA)

            def _v(x):
                return None if pd.isna(x) else float(x)

            sm.update(
                temp_c=_v(temp),
                snowfall_mm=_v(snowfall),
                snow_depth_cm=_v(depth),
                rain_mm=_v(rain),
                irr_wm2=_v(irr),
                prev_snow_depth_cm=_v(prev_depth),
            )
            prev_depth = depth
            factors.append(sm.factor())

        df["snow_factor"] = factors
        df["pv_kw_pred"] = pd.to_numeric(df["pv_kw_pred"], errors="coerce").fillna(0.0) * df["snow_factor"]
    except Exception:
        df["snow_factor"] = 1.0

    return df


# ---------------------------
# Switch window logic (existing)
# ---------------------------

def _elapsed_month_off_allowance_hours(run_day: date, allowance_frac: float) -> float:
    return max(0.0, float(allowance_frac)) * float(run_day.day) * 24.0


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


def _resolve_used_off_hours(
    run_day: date,
    summary_existing: pd.DataFrame | None,
) -> tuple[float | None, str]:
    """
    Prefer live availability-derived OFF hours, but fall back to the summary CSV
    when telemetry is unavailable. This keeps switching functional in offline /
    partial environments while still honoring the monthly budget.
    """
    used = _used_off_hours_from_availability(run_day)
    if used is not None:
        return float(used), "availability"

    month_key = f"{run_day.year:04d}-{run_day.month:02d}"
    return float(_compute_used_off_hours(summary_existing, month_key)), "summary_csv"


def _used_off_hours_from_availability(run_day: date) -> float | None:
    """
    Estimate used OFF hours from DisponibilitaPripojeni (dataPointPercentage).
    dataPointPercentage is ON availability for current month so far.
    """
    try:
        from monitor.availability_check import fetch_current_month_availability

        sample = fetch_current_month_availability()
        if (sample.month_start_local.year, sample.month_start_local.month) != (run_day.year, run_day.month):
            return None

        on_frac = max(0.0, min(1.0, float(sample.dispon_pripojeni)))
        off_frac = 1.0 - on_frac
        elapsed_hours = float(run_day.day) * 24.0
        return max(0.0, elapsed_hours * off_frac)
    except Exception:
        return None


def _simulate_day(
    pv_kw: list[float],
    hours: list[int],
    off_start_idx: int | None,
    off_len: int,
    cfg: Config,
    soc_start_pct: float,
) -> dict[str, Any]:
    cap = float(cfg.battery_kwh_total) * float(cfg.battery_usable_frac)
    cap = max(0.1, cap)

    soc0 = max(0.0, min(100.0, float(soc_start_pct)))
    E = (soc0 / 100.0) * cap

    export_kwh = 0.0
    spill_kwh = 0.0
    first_full_hour: Optional[int] = None
    soc_at_off_start_frac: Optional[float] = None

    off_end_idx = None
    if off_start_idx is not None and off_len > 0:
        off_end_idx = off_start_idx + off_len

    for i, pv in enumerate(pv_kw):
        h = hours[i]
        pv = max(0.0, float(pv))
        if off_start_idx is not None and i == off_start_idx and soc_at_off_start_frac is None:
            soc_at_off_start_frac = E / cap

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
        "soc_at_off_start_frac": soc_at_off_start_frac,
    }


def _simulate_steps(
    pv_kw: list[float],
    step_times: list[pd.Timestamp],
    off_start_idx: int | None,
    off_len: int,
    cfg: Config,
    soc_start_pct: float,
    step_hours: float,
) -> dict[str, Any]:
    cap = float(cfg.battery_kwh_total) * float(cfg.battery_usable_frac)
    cap = max(0.1, cap)
    dt_h = max(0.01, float(step_hours))

    soc0 = max(0.0, min(100.0, float(soc_start_pct)))
    E = (soc0 / 100.0) * cap

    export_kwh = 0.0
    spill_kwh = 0.0
    first_full_hour: Optional[int] = None
    soc_at_off_start_frac: Optional[float] = None

    off_end_idx = None
    if off_start_idx is not None and off_len > 0:
        off_end_idx = off_start_idx + off_len

    for i, pv in enumerate(pv_kw):
        pv = max(0.0, float(pv))
        if off_start_idx is not None and i == off_start_idx and soc_at_off_start_frac is None:
            soc_at_off_start_frac = E / cap

        after_load_kw = max(0.0, pv - float(cfg.baseload_kw))
        in_off = False
        if off_start_idx is not None and off_end_idx is not None:
            in_off = (i >= off_start_idx) and (i < off_end_idx)

        batt_rem = max(0.0, cap - E)
        batt_rem_kw_equiv = batt_rem / dt_h

        if in_off:
            export_kw = min(float(cfg.export_limit_kw), after_load_kw)
            rem_kw = max(0.0, after_load_kw - export_kw)
            charge_kw = min(rem_kw, batt_rem_kw_equiv)
            spill_kw = max(0.0, rem_kw - charge_kw)
        else:
            charge_kw = min(after_load_kw, batt_rem_kw_equiv)
            rem_kw = max(0.0, after_load_kw - charge_kw)
            if batt_rem <= 1e-9:
                export_kw = min(float(cfg.export_limit_kw), after_load_kw)
                spill_kw = max(0.0, after_load_kw - export_kw)
            else:
                export_kw = 0.0
                spill_kw = rem_kw

        E = min(cap, E + charge_kw * dt_h)
        export_kwh += export_kw * dt_h
        spill_kwh += spill_kw * dt_h

        if first_full_hour is None and E >= cap - 1e-6:
            first_full_hour = int(pd.Timestamp(step_times[i]).hour)

    return {
        "export_kwh": export_kwh,
        "spill_kwh": spill_kwh,
        "soc_end_frac": E / cap,
        "first_full_hour": first_full_hour,
        "soc_at_off_start_frac": soc_at_off_start_frac,
    }


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = idx
        prev = idx
    runs.append((start, prev))
    return runs


def _hour_to_index(hours: list[int], target_hour: int) -> int | None:
    for i, h in enumerate(hours):
        if int(h) >= int(target_hour):
            return i
    return None


def fetch_spot_prices_qh(cfg: Config, run_day: date) -> pd.DataFrame | None:
    if not bool(cfg.spot_price_enabled):
        return None
    if run_day != date.today():
        return None

    try:
        resp = requests.get(str(cfg.spot_price_api_url), timeout=8)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return None

    rows = payload.get("hoursToday")
    if not isinstance(rows, list) or not rows:
        return None

    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            hh = int(row.get("hour"))
            mm = int(row.get("minute", 0))
            ts = datetime.combine(run_day, dtime(hh, mm))
            parsed.append({
                "time": pd.Timestamp(ts),
                "price_czk": float(row.get("priceCZK")),
                "price_eur": float(row.get("priceEur")) if row.get("priceEur") is not None else pd.NA,
                "price_level": row.get("level"),
                "price_level_num96": row.get("levelNum96"),
            })
        except Exception:
            continue

    if not parsed:
        return None

    df = pd.DataFrame(parsed).sort_values("time").drop_duplicates(subset=["time"], keep="last")
    return df.reset_index(drop=True)


def _expand_hourly_to_quarter_hour(today: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in today.iterrows():
        base_time = pd.Timestamp(row["time"])
        for minute in (0, 15, 30, 45):
            item = row.to_dict()
            item["time"] = base_time + timedelta(minutes=minute)
            rows.append(item)
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def recommend_switch_window_smart(
    hourly_df: pd.DataFrame,
    run_day: date,
    pred_today_kwh: float,
    cfg: Config,
    summary_existing: pd.DataFrame | None,
    soc_start_pct: float,
) -> Tuple[str, str, dict[str, Any]]:
    trace: dict[str, Any] = {
        "run_day": run_day.isoformat(),
        "pred_today_kwh": float(pred_today_kwh),
        "soc_start_pct": float(soc_start_pct),
        "min_pred_kwh_for_switch": float(cfg.min_pred_kwh_for_switch),
        "min_peak_kw_for_switch": float(cfg.min_peak_kw_for_switch),
        "monthly_off_allowance_frac": float(cfg.monthly_off_allowance_frac),
        "max_off_hours": float(cfg.max_off_hours),
        "preferred_max_off_hours": float(cfg.preferred_max_off_hours),
        "preferred_switch_start_hour": int(cfg.preferred_switch_start_hour),
        "search_window_hours": [int(cfg.switch_search_start_hour), int(cfg.switch_search_end_hour)],
        "spot_price_enabled": bool(cfg.spot_price_enabled),
        "spot_price_negative_threshold_czk": float(cfg.spot_price_negative_threshold_czk),
        "spot_price_pv_score_tolerance": float(cfg.spot_price_pv_score_tolerance),
    }
    if float(pred_today_kwh) < float(cfg.min_pred_kwh_for_switch):
        trace["decision"] = "no_switch"
        trace["reason"] = "pred_today_below_threshold"
        return ("", "", trace)

    today = hourly_df[hourly_df["date"] == run_day].copy()
    if today.empty:
        trace["decision"] = "no_switch"
        trace["reason"] = "no_hourly_rows_for_day"
        return ("", "", trace)

    today = today.sort_values("time")
    trace["hourly_points"] = int(len(today))
    use_price_qh = False
    steps = today.copy()
    step_hours = 1.0
    spot_prices = fetch_spot_prices_qh(cfg, run_day)
    if spot_prices is not None and not spot_prices.empty:
        qh = _expand_hourly_to_quarter_hour(today)
        steps = qh.merge(spot_prices, on="time", how="left")
        matched_prices = pd.to_numeric(steps.get("price_czk"), errors="coerce").notna().sum()
        if matched_prices >= max(80, len(steps) - 8):
            use_price_qh = True
            step_hours = 0.25
        else:
            steps = today.copy()

    trace["switch_resolution_minutes"] = int(round(step_hours * 60.0))
    trace["spot_price_mode"] = "quarter_hour" if use_price_qh else "disabled_or_unavailable"
    if use_price_qh:
        price_series = pd.to_numeric(steps["price_czk"], errors="coerce")
        trace["spot_price_points"] = int(price_series.notna().sum())
        trace["spot_price_negative_points"] = int((price_series < float(cfg.spot_price_negative_threshold_czk)).sum())
        if price_series.notna().any():
            trace["spot_price_min_czk"] = float(price_series.min())
            trace["spot_price_avg_czk"] = float(price_series.mean())

    pv = pd.to_numeric(steps["pv_kw_pred"], errors="coerce").fillna(0.0).tolist()
    step_times = [pd.Timestamp(t) for t in steps["time"].tolist()]
    hours = [int(ts.hour) for ts in step_times]
    if len(pv) < 8:
        trace["decision"] = "no_switch"
        trace["reason"] = "insufficient_hourly_points"
        return ("", "", trace)

    year, month = run_day.year, run_day.month
    month_key = f"{year:04d}-{month:02d}"
    used, used_source = _resolve_used_off_hours(run_day, summary_existing)
    allowance_so_far = _elapsed_month_off_allowance_hours(run_day, cfg.monthly_off_allowance_frac)
    remaining_allowance_so_far = max(0.0, allowance_so_far - used)
    trace["month_key"] = month_key
    trace["used_off_hours_source"] = used_source
    trace["used_off_hours_so_far"] = float(used)
    trace["allowance_so_far_hours"] = float(allowance_so_far)
    trace["remaining_allowance_so_far_hours"] = float(remaining_allowance_so_far)

    # allocate more budget to higher-than-average days (within month)
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

    trace["month_mean_pred_kwh"] = (None if month_mean_pred is None else float(month_mean_pred))
    trace["weight"] = float(weight)

    alloc_today = min(
        float(cfg.max_off_hours),
        float(cfg.preferred_max_off_hours),
        remaining_allowance_so_far,
        remaining_allowance_so_far * weight,
    )
    trace["alloc_today_hours"] = float(alloc_today)
    if alloc_today < 0.75:
        trace["decision"] = "no_switch"
        trace["reason"] = "alloc_below_minimum"
        return ("", "", trace)

    export_trigger_kw = max(
        float(cfg.export_limit_kw) + float(cfg.baseload_kw),
        float(cfg.min_peak_kw_for_switch),
    )
    over_trigger_idx = [i for i, pv_kw in enumerate(pv) if float(pv_kw) >= export_trigger_kw]
    over_trigger_runs = _contiguous_runs(over_trigger_idx)
    trace["export_trigger_kw"] = float(export_trigger_kw)
    trace["hours_over_trigger"] = [int(hours[i]) for i in over_trigger_idx]
    trace["over_trigger_runs"] = [[int(hours[s]), int(hours[e])] for s, e in over_trigger_runs]
    trace["peak_pv_kw"] = float(max(pv) if pv else 0.0)
    peak_idx = (None if not pv else int(pv.index(max(pv))))
    peak_hour = (None if peak_idx is None else int(hours[peak_idx]))
    trace["peak_hour"] = peak_hour
    if not over_trigger_idx:
        trace["decision"] = "no_switch"
        trace["reason"] = "no_hours_over_export_trigger"
        return ("", "", trace)

    min_candidate_steps = max(1, int(round(1.0 / step_hours)))
    maxL = max(min_candidate_steps, int(round(float(alloc_today) / step_hours)))
    trace["min_candidate_length_hours"] = float(min_candidate_steps * step_hours)
    trace["max_candidate_length_hours"] = float(maxL * step_hours)

    lead_hours = 0
    if float(pred_today_kwh) >= 32.0 or float(soc_start_pct) >= 50.0:
        lead_hours = 1
    if float(pred_today_kwh) >= 38.0 or float(soc_start_pct) >= 65.0:
        lead_hours = 2
    lead_steps = int(round(float(lead_hours) / step_hours))
    earliest_practical_start_hour = max(int(cfg.switch_search_start_hour), int(cfg.preferred_switch_start_hour) - 1)
    first_trigger_idx = over_trigger_idx[0]
    first_candidate_idx = max(0, first_trigger_idx - lead_steps)
    earliest_practical_idx = _hour_to_index(hours, earliest_practical_start_hour)
    if earliest_practical_idx is not None:
        first_candidate_idx = max(first_candidate_idx, earliest_practical_idx)

    start_h = max(int(cfg.switch_search_start_hour), earliest_practical_start_hour)
    end_h = int(cfg.switch_search_end_hour)
    candidates = [
        i for i in range(first_candidate_idx, len(hours))
        if start_h <= hours[i] <= end_h and i <= over_trigger_idx[-1]
    ]
    trace["lead_hours_before_trigger"] = int(lead_hours)
    trace["candidate_start_times"] = [step_times[i].strftime("%H:%M") for i in candidates[:96]]
    if not candidates:
        trace["decision"] = "no_switch"
        trace["reason"] = "no_candidate_hours_after_filters"
        return ("", "", trace)
    trace["candidate_start_count"] = len(candidates)

    sunset_cutoff = None
    if "sunset" in today.columns:
        sunset_values = today["sunset"].dropna()
        if not sunset_values.empty:
            sunset_cutoff = pd.Timestamp(sunset_values.iloc[0]) - timedelta(hours=2)
    trace["latest_switch_on_allowed"] = (
        None if sunset_cutoff is None else pd.Timestamp(sunset_cutoff).isoformat()
    )

    if use_price_qh:
        base = _simulate_steps(pv, step_times, None, 0, cfg, soc_start_pct, step_hours)
    else:
        base = _simulate_day(pv, hours, None, 0, cfg, soc_start_pct)

    def score(sim: dict[str, Any]) -> float:
        export = float(sim["export_kwh"])
        spill = float(sim["spill_kwh"])
        end_frac = float(sim["soc_end_frac"])
        full_hour = sim.get("first_full_hour", None)
        off_start_frac = sim.get("soc_at_off_start_frac", None)

        target = float(cfg.target_soc_evening)
        deficit = max(0.0, target - end_frac)
        penalty_end = 1500.0 * (deficit ** 2)

        penalty_early_full = 0.0
        if full_hour is not None and peak_hour is not None and int(full_hour) < int(peak_hour):
            penalty_early_full = 6.0 * float(int(peak_hour) - int(full_hour))

        penalty_low_soc_at_start = 0.0
        if off_start_frac is not None and peak_hour is not None:
            if float(off_start_frac) < 0.30:
                penalty_low_soc_at_start = 8.0 * float(0.30 - float(off_start_frac))

        return (3.5 * export) - (4.0 * spill) - penalty_end - penalty_early_full - penalty_low_soc_at_start

    base_score = score(base)
    trace["base"] = {
        "export_kwh": float(base["export_kwh"]),
        "spill_kwh": float(base["spill_kwh"]),
        "soc_end_frac": float(base["soc_end_frac"]),
        "soc_at_off_start_frac": base.get("soc_at_off_start_frac", None),
        "first_full_hour": base.get("first_full_hour", None),
        "score": float(base_score),
    }
    best = {"score": base_score, "off": "", "on": "", "L": 0, "sim": None}
    candidate_records: list[dict[str, Any]] = []

    for L in range(min_candidate_steps, maxL + 1):
        for s_idx in candidates:
            e_idx = s_idx + L
            if e_idx > len(pv):
                continue
            matches_run = any(
                (run_start - lead_steps) <= s_idx <= run_end and (e_idx - 1) <= run_end
                for run_start, run_end in over_trigger_runs
            )
            if not matches_run:
                continue
            if sunset_cutoff is not None:
                candidate_on = step_times[e_idx - 1] + timedelta(hours=step_hours)
                if pd.Timestamp(candidate_on) > pd.Timestamp(sunset_cutoff):
                    continue
            if use_price_qh:
                sim = _simulate_steps(pv, step_times, s_idx, L, cfg, soc_start_pct, step_hours)
            else:
                sim = _simulate_day(pv, hours, s_idx, L, cfg, soc_start_pct)
            sc = score(sim)
            off_time = step_times[s_idx].to_pydatetime().strftime("%H:%M")
            on_time = (step_times[e_idx - 1].to_pydatetime() + timedelta(hours=step_hours)).strftime("%H:%M")
            rec = {"score": sc, "off": off_time, "on": on_time, "L": L, "sim": sim, "s_idx": s_idx, "e_idx": e_idx}
            if use_price_qh and "price_czk" in steps.columns:
                prices = pd.to_numeric(steps.iloc[s_idx:e_idx]["price_czk"], errors="coerce").dropna()
                rec["price_avg_czk"] = (None if prices.empty else float(prices.mean()))
                rec["price_min_czk"] = (None if prices.empty else float(prices.min()))
                rec["price_negative_points"] = int(
                    (prices < float(cfg.spot_price_negative_threshold_czk)).sum()
                )
            candidate_records.append(rec)
            if sc > best["score"]:
                best = rec

    if use_price_qh and candidate_records:
        best_pv_score = max(float(rec["score"]) for rec in candidate_records)
        shortlist = [
            rec
            for rec in candidate_records
            if float(rec["score"]) >= (best_pv_score - float(cfg.spot_price_pv_score_tolerance))
        ]
        price_series = pd.to_numeric(steps.get("price_czk"), errors="coerce")
        negative_available = bool((price_series < float(cfg.spot_price_negative_threshold_czk)).any())
        trace["price_shortlist_size"] = int(len(shortlist))
        trace["price_negative_available"] = negative_available
        if shortlist:
            if negative_available:
                shortlist.sort(
                    key=lambda rec: (
                        -int(rec.get("price_negative_points", 0)),
                        float("inf") if rec.get("price_min_czk") is None else float(rec["price_min_czk"]),
                        float("inf") if rec.get("price_avg_czk") is None else float(rec["price_avg_czk"]),
                        -float(rec["score"]),
                        rec["off"],
                    )
                )
            else:
                shortlist.sort(
                    key=lambda rec: (
                        float("inf") if rec.get("price_avg_czk") is None else float(rec["price_avg_czk"]),
                        float("inf") if rec.get("price_min_czk") is None else float(rec["price_min_czk"]),
                        -float(rec["score"]),
                        rec["off"],
                    )
                )
            best = shortlist[0]

    if best["L"] == 0 or (best["score"] - base_score) < 1.0:
        trace["best"] = {
            "score": float(best["score"]),
            "off": str(best["off"]),
            "on": str(best["on"]),
            "length_hours": float(best["L"] * step_hours),
            "improvement_vs_base": float(best["score"] - base_score),
            "soc_at_off_start_frac": (
                None if best["sim"] is None else best["sim"].get("soc_at_off_start_frac", None)
            ),
            "soc_end_frac": None if best["sim"] is None else float(best["sim"]["soc_end_frac"]),
            "price_avg_czk": best.get("price_avg_czk", None),
            "price_min_czk": best.get("price_min_czk", None),
            "price_negative_points": best.get("price_negative_points", None),
        }
        trace["decision"] = "no_switch"
        trace["reason"] = "no_meaningful_improvement"
        return ("", "", trace)

    trace["best"] = {
        "score": float(best["score"]),
        "off": str(best["off"]),
        "on": str(best["on"]),
        "length_hours": float(best["L"] * step_hours),
        "improvement_vs_base": float(best["score"] - base_score),
        "soc_at_off_start_frac": (
            None if best["sim"] is None else best["sim"].get("soc_at_off_start_frac", None)
        ),
        "soc_end_frac": None if best["sim"] is None else float(best["sim"]["soc_end_frac"]),
        "price_avg_czk": best.get("price_avg_czk", None),
        "price_min_czk": best.get("price_min_czk", None),
        "price_negative_points": best.get("price_negative_points", None),
    }
    trace["decision"] = "switch"
    trace["reason"] = "best_candidate_selected"
    return (str(best["off"]), str(best["on"]), trace)


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
    df = apply_snow_correction(cfg, df)
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


def save_intraday_csv(df_hourly: pd.DataFrame, run_day: date) -> Path:
    out_dir = Path("forecasts/intraday")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"forecast_intraday_{run_day.isoformat()}.csv"

    today = df_hourly[df_hourly["date"] == run_day].copy().sort_values("time")
    today["step_kwh"] = pd.to_numeric(today["pv_kw_pred"], errors="coerce").fillna(0.0)  # 1h step => kWh == kW
    export = today.copy()
    export["time"] = export["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    export.to_csv(out_path, index=False)
    return out_path


def cleanup_intraday_files(keep_days: int = 30) -> int:
    d = Path("forecasts/intraday")
    if not d.exists():
        return 0
    cutoff = date.today() - timedelta(days=int(keep_days))
    deleted = 0
    for p in d.glob("forecast_intraday_*.csv"):
        m = re.search(r"forecast_intraday_(\d{4}-\d{2}-\d{2})\.csv$", p.name)
        if not m:
            continue
        try:
            dd = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except Exception:
            continue
        if dd < cutoff:
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
    if "snow_factor" in df.columns:
        # show snow factor lightly (scaled) for debugging
        try:
            sf = pd.to_numeric(df["snow_factor"], errors="coerce").fillna(1.0)
            plt.plot(df["time"], sf * df["pv_kw_pred"].max(), label="snow_factor (scaled)")
        except Exception:
            pass
    plt.title(f"PV forecast (hourly) – generated {run_day.isoformat()}")
    plt.xlabel("Time")
    plt.ylabel("kW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# ---------------------------
# Summary CSV upsert
# ---------------------------

def _round2(x: float) -> float:
    try:
        return float(round(float(x), 2))
    except Exception:
        return float("nan")


SUMMARY_COLUMNS = [
    "Date",
    "PredictionToday",
    "ActualToday",
    "PredictionTomorrow",
    "SwitchOff",
    "SwitchOn",
    "Enabled",
]


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

    new_row = pd.DataFrame([{
        "Date": run_day.isoformat(),
        "PredictionToday": _round2(pred_today),
        "ActualToday": (pd.NA if actual_today is None else _round2(actual_today)),
        "PredictionTomorrow": _round2(pred_tomorrow),
        "SwitchOff": (switch_off or ""),
        "SwitchOn": (switch_on or ""),
        # Keep switching enabled by default; blank is the compatibility/default state.
        "Enabled": "",
    }])

    if out_path.exists():
        old = pd.read_csv(out_path)
        for c in SUMMARY_COLUMNS:
            if c not in old.columns:
                old[c] = pd.NA
        old["Date"] = old["Date"].astype(str)
        old = old[old["Date"] != run_day.isoformat()]
        merged = pd.concat([old, new_row], ignore_index=True)
    else:
        merged = new_row

    merged = merged[SUMMARY_COLUMNS].sort_values("Date")
    merged.to_csv(out_path, index=False)
    return out_path


def save_switch_trace(run_day: date, trace: dict[str, Any]) -> Path:
    out_dir = Path("forecasts/switch_trace")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"switch_trace_{run_day.isoformat()}.json"
    out_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
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


# ---------------------------
# Main
# ---------------------------

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

    intraday_csv = save_intraday_csv(hourly, today)
    print(f"Intraday CSV uloženo: {intraday_csv.as_posix()}")

    deleted = cleanup_intraday_files(keep_days=30)
    if deleted:
        print(f"Intraday cleanup: deleted {deleted} old file(s)")

    summary_path = Path(args.out_summary)
    existing = _load_existing_summary(summary_path)

    sw_off, sw_on, sw_trace = recommend_switch_window_smart(
        hourly_df=hourly,
        run_day=today,
        pred_today_kwh=today_kwh,
        cfg=cfg,
        summary_existing=existing,
        soc_start_pct=soc_pct,
    )
    trace_path = save_switch_trace(today, sw_trace)
    print(f"Switch trace uložena: {trace_path.as_posix()}")

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
