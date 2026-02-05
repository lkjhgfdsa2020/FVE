#!/usr/bin/env python3
"""
pv_forecast.py

Forecast expected PV production (hourly curve) for today + tomorrow.

- Open-Meteo forecast hourly global_tilted_irradiance (GTI) for your tilt/azimuth
- Monthly PR (performance ratio) from pr_calendar.json (optional; fallback to config.json)
- Optional cap from historical clear-day envelope built from SolaX exports in data/solax/*.xlsx

Outputs:
- Console: predicted daily kWh for today and tomorrow (+ PR used)
- forecast_outputs/pv_hourly_forecast_<YYYY-MM-DD>.csv
- forecast_plots/pv_forecast_<YYYY-MM-DD>.png
- forecast_outputs/forecast_daily_summary.csv (Date,PredictionToday,PredictionTomorrow) - upsert per run

Cache:
- cache/clear_envelope.parquet (preferred; needs pyarrow)
- cache/clear_envelope.csv (fallback)

"""

import glob
import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solax_io import read_solax_excel


# ---------- Config ----------
@dataclass
class Config:
    latitude: float
    longitude: float
    timezone: str

    pv_kwp: float
    tilt_deg: float
    azimuth_deg_from_north: float

    export_limit_kw: float
    baseload_kw: float
    max_off_hours: int

    performance_ratio: float
    soc_8am: float


def load_config(path: str = "config.json") -> Config:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return Config(**cfg)


def load_pr_calendar(path: str = "pr_calendar.json") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in data.items()}
    except Exception:
        return {}


def pr_for_month(pr_map: dict, month_key: str, fallback: float) -> float:
    v = pr_map.get(month_key)
    if v is None:
        return float(fallback)
    # sensible clamp
    return float(max(0.5, min(1.0, v)))


def azimuth_to_openmeteo(az_from_north: float) -> float:
    """
    User azimuth: degrees from north clockwise (0=N, 90=E, 180=S, 270=W).
    Open-Meteo: 0=south, -90=east, +90=west, ±180=north.
    Conversion: openmeteo = az_from_north - 180
    """
    return float(az_from_north) - 180.0


# ---------- SolaX history -> clear-day envelope ----------
def build_clear_envelope(solax_glob: str = "data/solax/*.xlsx") -> pd.DataFrame:
    paths = sorted(glob.glob(solax_glob))
    if not paths:
        raise FileNotFoundError(
            f"Nenašel jsem žádné SolaX exporty podle '{solax_glob}'. Dej XLSX exporty do data/solax/."
        )

    df_all = pd.concat([read_solax_excel(p) for p in paths], ignore_index=True)
    df_all = df_all.sort_values("Update time").set_index("Update time")

    # hourly mean PV power (kW)
    pv_kw_h = (df_all["Total PV Power (W)"].resample("h").mean() / 1000.0).rename("pv_kw")

    # daily energy (kWh)
    pv_kwh_d = pv_kw_h.resample("D").sum(min_count=1).rename("pv_kwh")

    daily = pv_kwh_d.reset_index().rename(columns={"Update time": "day"})
    daily["month"] = daily["day"].dt.to_period("M").astype(str)
    daily["date"] = daily["day"].dt.date
    daily["rank"] = daily.groupby("month")["pv_kwh"].rank(pct=True)

    # top 10% days per month ~ clear days
    clear_days = set(daily.loc[daily["rank"] >= 0.9, "date"])

    hour = pv_kw_h.reset_index().rename(columns={"Update time": "time"})
    hour["date"] = hour["time"].dt.date
    hour["month"] = hour["time"].dt.to_period("M").astype(str)
    hour["hour"] = hour["time"].dt.hour

    env = (
        hour[hour["date"].isin(clear_days)]
        .groupby(["month", "hour"])["pv_kw"]
        .median()
        .rename("pv_kw_clear_env")
        .reset_index()
    )
    return env


def _read_envelope_cache(parquet_path: Path, csv_path: Path) -> pd.DataFrame | None:
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    return None


def _write_envelope_cache(env: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    # try parquet first; fallback to csv
    try:
        env.to_parquet(parquet_path, index=False)
        return
    except Exception:
        env.to_csv(csv_path, index=False)


def load_or_build_envelope(solax_glob: str = "data/solax/*.xlsx") -> pd.DataFrame:
    cache_dir = Path("cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = cache_dir / "clear_envelope.parquet"
    csv_path = cache_dir / "clear_envelope.csv"

    cached = _read_envelope_cache(parquet_path, csv_path)
    if cached is not None and not cached.empty:
        return cached

    env = build_clear_envelope(solax_glob=solax_glob)
    _write_envelope_cache(env, parquet_path, csv_path)
    return env


# ---------- Open-Meteo forecast ----------
def fetch_open_meteo_forecast(cfg: Config, days: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_az = azimuth_to_openmeteo(cfg.azimuth_deg_from_north)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={cfg.latitude}"
        f"&longitude={cfg.longitude}"
        f"&timezone={cfg.timezone}"
        f"&forecast_days={days}"
        f"&tilt={cfg.tilt_deg}"
        f"&azimuth={panel_az}"
        "&hourly=global_tilted_irradiance"
        "&daily=sunrise,sunset"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()

    hourly = pd.DataFrame(
        {
            "time": pd.to_datetime(j["hourly"]["time"]),
            "gti": pd.to_numeric(j["hourly"]["global_tilted_irradiance"], errors="coerce"),
        }
    )
    hourly["date"] = hourly["time"].dt.date
    hourly["hour"] = hourly["time"].dt.hour
    hourly["month"] = hourly["time"].dt.to_period("M").astype(str)

    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(j["daily"]["time"])).dt.date,
            "sunrise": pd.to_datetime(pd.Series(j["daily"]["sunrise"])),
            "sunset": pd.to_datetime(pd.Series(j["daily"]["sunset"])),
        }
    )
    return hourly, daily


def pv_from_gti_kw(pv_kwp: float, pr: float, gti_wm2: float) -> float:
    if gti_wm2 is None or (isinstance(gti_wm2, float) and math.isnan(gti_wm2)):
        return 0.0
    pv_kw = pv_kwp * pr * (float(gti_wm2) / 1000.0)
    return float(max(0.0, min(pv_kwp, pv_kw)))


def make_hourly_forecast(cfg: Config, pr_map: dict, env: pd.DataFrame, start_day: date, days: int = 2) -> pd.DataFrame:
    hourly, daily = fetch_open_meteo_forecast(cfg, days=days)
    daily_map = daily.set_index("date")[["sunrise", "sunset"]].to_dict("index")

    wanted_days = [start_day + timedelta(d) for d in range(days)]
    h = hourly[hourly["date"].isin(wanted_days)].copy()

    # PR per row/month
    h["month_key"] = h["time"].dt.strftime("%Y-%m")
    h["pr_used"] = h["month_key"].apply(lambda mk: pr_for_month(pr_map, mk, cfg.performance_ratio))

    # base model from GTI
    h["pv_kw_gti"] = h.apply(lambda r: pv_from_gti_kw(cfg.pv_kwp, r["pr_used"], r["gti"]), axis=1)

    # optional cap from clear-day envelope
    h = h.merge(env, on=["month", "hour"], how="left")
    h["pv_kw_clear_env"] = h["pv_kw_clear_env"].fillna(cfg.pv_kwp)
    h["pv_kw_pred"] = h[["pv_kw_gti", "pv_kw_clear_env"]].min(axis=1)

    # zero outside daylight window
    def in_daylight(row) -> bool:
        info = daily_map.get(row["date"])
        if not info:
            return True
        return (row["time"] >= info["sunrise"]) and (row["time"] <= info["sunset"])

    mask = h.apply(in_daylight, axis=1)
    h.loc[~mask, "pv_kw_pred"] = 0.0
    h.loc[~mask, "pv_kw_gti"] = 0.0

    return h.sort_values("time").reset_index(drop=True)


def plot_today_tomorrow(df: pd.DataFrame, start_day: date) -> Path:
    days = sorted(df["date"].unique())
    plt.figure(figsize=(11, 5))
    for d in days:
        ddf = df[df["date"] == d].copy()
        plt.plot(ddf["time"], ddf["pv_kw_pred"], label=str(d))

    plt.title("Očekávaný PV výkon (hodinově) – dnes + zítra")
    plt.xlabel("Čas")
    plt.ylabel("Výkon (kW)")
    plt.legend()
    plt.tight_layout()

    out = Path("forecast_plots")
    out.mkdir(exist_ok=True)
    fn = out / f"pv_forecast_{start_day.isoformat()}.png"
    plt.savefig(fn, dpi=160)
    plt.close()
    return fn


def upsert_daily_forecast_summary(
    run_day: date,
    pred_today: float,
    pred_tomorrow: float,
    out_csv: str = "forecast_outputs/forecast_daily_summary.csv",
) -> Path:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame(
        [{
            "Date": run_day.isoformat(),
            "PredictionToday": float(pred_today),
            "PredictionTomorrow": float(pred_tomorrow),
        }]
    )

    if out_path.exists():
        old = pd.read_csv(out_path)
        # ensure required columns exist
        for c in ["Date", "PredictionToday", "PredictionTomorrow"]:
            if c not in old.columns:
                old[c] = pd.NA
        old["Date"] = old["Date"].astype(str)

        # upsert by Date
        old = old[old["Date"] != run_day.isoformat()]
        merged = pd.concat([old, new_row], ignore_index=True)
    else:
        merged = new_row

    merged = merged.sort_values("Date")
    merged.to_csv(out_path, index=False)
    return out_path


def main():
    cfg = load_config("config.json")
    pr_map = load_pr_calendar("pr_calendar.json")

    env = load_or_build_envelope(solax_glob="data/solax/*.xlsx")

    today = date.today()
    tomorrow = today + timedelta(days=1)

    df = make_hourly_forecast(cfg, pr_map, env, today, days=2)

    summary = df.groupby("date")["pv_kw_pred"].sum().rename("pv_kwh_pred").reset_index()
    print("\nPredikce denní výroby (kWh):")
    print(summary.to_string(index=False))

    pr_used = df.groupby("date")["pr_used"].median().rename("pr_used").reset_index()
    if not pr_used.empty:
        print("\nPoužité PR (median za den):")
        print(pr_used.to_string(index=False))

    # Save hourly CSV
    out = Path("forecast_outputs")
    out.mkdir(exist_ok=True)
    hourly_csv = out / f"pv_hourly_forecast_{today.isoformat()}.csv"
    df.to_csv(hourly_csv, index=False)
    print(f"CSV uloženo: {hourly_csv}")

    # Plot
    plot_path = plot_today_tomorrow(df, today)
    print(f"Graf uložen: {plot_path}")

    # Save daily summary CSV (Date,PredictionToday,PredictionTomorrow)
    today_kwh = float(summary.loc[summary["date"] == today, "pv_kwh_pred"].iloc[0]) if (summary["date"] == today).any() else 0.0
    tomorrow_kwh = float(summary.loc[summary["date"] == tomorrow, "pv_kwh_pred"].iloc[0]) if (summary["date"] == tomorrow).any() else 0.0

    summary_path = upsert_daily_forecast_summary(
        run_day=today,
        pred_today=today_kwh,
        pred_tomorrow=tomorrow_kwh,
        out_csv="forecasts/forecast_daily_summary.csv",
    )
    print(f"Souhrn uložen/aktualizován: {summary_path}")


if __name__ == "__main__":
    main()
