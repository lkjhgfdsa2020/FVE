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
# Date,PredictionToday,ActualToday,PredictionTomorrow,SwitchOff,SwitchOn
#
# Snow correction (optional):
# - Fetches additional Open-Meteo hourly vars: temperature_2m, rain, snowfall, snow_depth
# - Applies multiplicative snow_factor per hour using snow_runtime.SnowModel
# - Exports snow_factor + snow meteo columns into hourly/intraday CSV for debugging
#
# Notes:
# - SwitchOff/SwitchOn are empty if recommendation is "do not switch off".
# - SoC start is fetched from SolaXCloud API if env vars exist; otherwise defaults to soc_default_pct.

from __future__ import annotations

import argparse
import calendar
import json
import os
import re
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

    max_off_hours: float = 10.0
    switch_search_start_hour: int = 8
    switch_search_end_hour: int = 16
    min_pred_kwh_for_switch: float = 20.0

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
        max_off_hours=float(j.get("max_off_hours", 10.0)),
        switch_search_start_hour=int(j.get("switch_search_start_hour", 8)),
        switch_search_end_hour=int(j.get("switch_search_end_hour", 16)),
        min_pred_kwh_for_switch=float(j.get("min_pred_kwh_for_switch", 20.0)),
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

def _month_off_budget_hours(year: int, month: int) -> float:
    days_in_month = calendar.monthrange(year, month)[1]
    return 0.20 * days_in_month * 24.0


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
) -> Tuple[str, str]:
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

    base_alloc = remaining_budget / remaining_days
    alloc_today = min(float(cfg.max_off_hours), base_alloc * weight, remaining_budget)
    if alloc_today < 0.75:
        return ("", "")

    maxL = max(1, int(round(alloc_today)))

    start_h = int(cfg.switch_search_start_hour)
    end_h = int(cfg.switch_search_end_hour)
    candidates = [i for i, h in enumerate(hours) if start_h <= h <= end_h]
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


if __name__ == "__main__":
    main()
