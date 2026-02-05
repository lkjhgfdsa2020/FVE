#!/usr/bin/env python3
"""
backtest_month.py (updated)

Backtest PV prediction vs. reality for a given month.

- Robust XLSX reading via solax_io.read_solax_excel() (handles varying sheet names/columns)
- PR fit optimized for DAILY kWh (energy fit)

Usage:
  python -u backtest_month.py --month 2024-11
  python -u backtest_month.py --month 2026-02
"""

import argparse
import calendar
import glob
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solax_io import read_solax_excel


@dataclass
class Config:
    latitude: float
    longitude: float
    timezone: str
    pv_kwp: float
    tilt_deg: float
    azimuth_deg_from_north: float
    performance_ratio: float


def load_config(path: str = "config.json") -> Config:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return Config(
        latitude=cfg["latitude"],
        longitude=cfg["longitude"],
        timezone=cfg["timezone"],
        pv_kwp=cfg["pv_kwp"],
        tilt_deg=cfg["tilt_deg"],
        azimuth_deg_from_north=cfg["azimuth_deg_from_north"],
        performance_ratio=cfg["performance_ratio"],
    )


def azimuth_to_openmeteo(az_from_north: float) -> float:
    return float(az_from_north) - 180.0


def load_solax_all(solax_glob: str) -> pd.DataFrame:
    paths = sorted(glob.glob(solax_glob))
    if not paths:
        raise FileNotFoundError(f"Nenašel jsem žádné SolaX exporty podle '{solax_glob}'.")
    frames = [read_solax_excel(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("Update time").set_index("Update time")


def solax_hourly_actual(df_solax: pd.DataFrame, start_ts: str, end_ts: str) -> pd.DataFrame:
    df = df_solax.loc[start_ts:end_ts].copy()
    pv_kw_h = (df["Total PV Power (W)"].resample("h").mean() / 1000.0).rename("pv_kw_actual")
    out = pv_kw_h.reset_index().rename(columns={"Update time": "time"})
    out["date"] = out["time"].dt.date
    out["hour"] = out["time"].dt.hour
    return out


def fetch_open_meteo_archive(cfg: Config, start: date, end: date, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"openmeteo_archive_{start.isoformat()}_{end.isoformat()}.json"

    if cache_file.exists():
        j = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        panel_az = azimuth_to_openmeteo(cfg.azimuth_deg_from_north)
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={cfg.latitude}"
            f"&longitude={cfg.longitude}"
            f"&start_date={start.isoformat()}"
            f"&end_date={end.isoformat()}"
            f"&timezone={cfg.timezone}"
            f"&tilt={cfg.tilt_deg}"
            f"&azimuth={panel_az}"
            "&hourly=global_tilted_irradiance"
            "&daily=sunrise,sunset"
        )
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        j = r.json()
        cache_file.write_text(json.dumps(j), encoding="utf-8")

    hourly = pd.DataFrame(
        {
            "time": pd.to_datetime(j["hourly"]["time"]),
            "gti": pd.to_numeric(j["hourly"]["global_tilted_irradiance"], errors="coerce"),
        }
    )
    hourly["date"] = hourly["time"].dt.date
    hourly["hour"] = hourly["time"].dt.hour

    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(j["daily"]["time"])).dt.date,
            "sunrise": pd.to_datetime(pd.Series(j["daily"]["sunrise"])),
            "sunset": pd.to_datetime(pd.Series(j["daily"]["sunset"])),
        }
    )
    return hourly, daily


def pv_from_gti(cfg: Config, gti_wm2: float, pr: float) -> float:
    if gti_wm2 is None or (isinstance(gti_wm2, float) and math.isnan(gti_wm2)):
        return 0.0
    pv_kw = cfg.pv_kwp * pr * (float(gti_wm2) / 1000.0)
    return max(0.0, min(cfg.pv_kwp, pv_kw))


def apply_daylight_zeroing(df_hourly: pd.DataFrame, df_daily: pd.DataFrame) -> pd.DataFrame:
    m = df_daily.set_index("date")[["sunrise", "sunset"]].to_dict("index")

    def is_daylight(row):
        info = m.get(row["date"])
        if not info:
            return True
        return (row["time"] >= info["sunrise"]) and (row["time"] <= info["sunset"])

    mask = df_hourly.apply(is_daylight, axis=1)
    df_hourly.loc[~mask, "pv_kw_pred"] = 0.0
    return df_hourly


def metrics_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("date").agg(
        pv_kwh_actual=("pv_kw_actual", "sum"),
        pv_kwh_pred=("pv_kw_pred", "sum"),
    ).reset_index()
    daily["err_kwh"] = daily["pv_kwh_pred"] - daily["pv_kwh_actual"]
    daily["abs_err_kwh"] = daily["err_kwh"].abs()
    denom = daily["pv_kwh_actual"].replace(0, pd.NA)
    daily["ape_%"] = (daily["abs_err_kwh"] / denom) * 100.0
    return daily


def fit_pr_daily_energy(cfg: Config, df_hourly: pd.DataFrame) -> float:
    base_kw = cfg.pv_kwp * (df_hourly["gti"] / 1000.0)
    tmp = df_hourly.copy()
    tmp["base_kw"] = base_kw

    daily = tmp.groupby("date").agg(
        e_actual=("pv_kw_actual", "sum"),
        e_base=("base_kw", "sum"),
    ).reset_index()

    denom = float(daily["e_base"].sum())
    if denom <= 0 or math.isnan(denom):
        return cfg.performance_ratio

    pr = float(daily["e_actual"].sum() / denom)
    return max(0.5, min(1.0, pr))


def plot_daily(daily: pd.DataFrame, title: str, out_path: Path):
    plt.figure(figsize=(12, 5))
    plt.plot(daily["date"], daily["pv_kwh_actual"], label="Skutečnost (kWh)")
    plt.plot(daily["date"], daily["pv_kwh_pred"], label="Predikce (kWh)")
    plt.title(title)
    plt.xlabel("Datum")
    plt.ylabel("kWh / den")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def parse_month(s: str) -> tuple[int, int]:
    parts = s.strip().split("-")
    if len(parts) != 2:
        raise ValueError("--month musí být ve formátu YYYY-MM (např. 2025-03)")
    y = int(parts[0])
    m = int(parts[1])
    if not (1 <= m <= 12):
        raise ValueError("Měsíc musí být 1..12")
    return y, m


def month_range(y: int, m: int) -> tuple[date, date]:
    last_day = calendar.monthrange(y, m)[1]
    return date(y, m, 1), date(y, m, last_day)


def run_backtest(cfg: Config, solax_glob: str, month_str: str) -> None:
    y, m = parse_month(month_str)
    start, end = month_range(y, m)

    solax = load_solax_all(solax_glob)
    actual_h = solax_hourly_actual(solax, f"{start.isoformat()}", f"{end.isoformat()} 23:59:59")

    cache_dir = Path("cache") / "openmeteo"
    weather_h, weather_d = fetch_open_meteo_archive(cfg, start, end, cache_dir=cache_dir)

    df = actual_h.merge(weather_h[["time", "gti"]], on="time", how="inner")

    pr0 = cfg.performance_ratio
    df["pv_kw_pred"] = df["gti"].apply(lambda g: pv_from_gti(cfg, g, pr0))
    df = apply_daylight_zeroing(df, weather_d)

    daily0 = metrics_daily(df)
    mae0 = float(daily0["abs_err_kwh"].mean())
    mape0 = float(daily0["ape_%"].dropna().mean()) if daily0["ape_%"].dropna().size else float("nan")

    pr_fit = fit_pr_daily_energy(cfg, df)
    df["pv_kw_pred"] = df["gti"].apply(lambda g: pv_from_gti(cfg, g, pr_fit))
    df = apply_daylight_zeroing(df, weather_d)

    daily1 = metrics_daily(df)
    mae1 = float(daily1["abs_err_kwh"].mean())
    mape1 = float(daily1["ape_%"].dropna().mean()) if daily1["ape_%"].dropna().size else float("nan")

    out_dir = Path("backtest_outputs") / f"{y:04d}-{m:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    daily0.to_csv(out_dir / "daily_pr_config.csv", index=False)
    daily1.to_csv(out_dir / "daily_pr_fit.csv", index=False)
    df.to_csv(out_dir / "hourly_merged.csv", index=False)

    plot_daily(daily0, f"{y:04d}-{m:02d} – skutečnost vs predikce (PR z configu)", out_dir / "daily_pr_config.png")
    plot_daily(daily1, f"{y:04d}-{m:02d} – skutečnost vs predikce (PR fit - daily energy)", out_dir / "daily_pr_fit.png")

    summary = {
        "month": f"{y:04d}-{m:02d}",
        "range": {"start": start.isoformat(), "end": end.isoformat()},
        "rows": {"hours_actual": int(len(actual_h)), "hours_merged": int(len(df))},
        "pr_config": float(pr0),
        "mae_kwh_per_day_pr_config": float(mae0),
        "mape_percent_pr_config": float(mape0) if not math.isnan(mape0) else None,
        "pr_fit_daily_energy": float(pr_fit),
        "mae_kwh_per_day_pr_fit": float(mae1),
        "mape_percent_pr_fit": float(mape1) if not math.isnan(mape1) else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Backtest summary ===")
    print(f"Month: {summary['month']} ({summary['range']['start']} → {summary['range']['end']})")
    print(f"Hours: actual={summary['rows']['hours_actual']}, merged={summary['rows']['hours_merged']}")
    print(
        f"PR(config)={summary['pr_config']:.3f} | "
        f"MAE={summary['mae_kwh_per_day_pr_config']:.2f} kWh/den | "
        f"MAPE={summary['mape_percent_pr_config'] if summary['mape_percent_pr_config'] is not None else 'n/a'}"
    )
    print(
        f"PR(fit)   ={summary['pr_fit_daily_energy']:.3f} | "
        f"MAE={summary['mae_kwh_per_day_pr_fit']:.2f} kWh/den | "
        f"MAPE={summary['mape_percent_pr_fit'] if summary['mape_percent_pr_fit'] is not None else 'n/a'}"
    )
    print(f"Saved to: {out_dir.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, help="Month in YYYY-MM, e.g. 2025-03")
    ap.add_argument("--config", default="config.json", help="Path to config.json")
    ap.add_argument("--solax-glob", default="data/solax/*.xlsx", help="Glob for SolaX XLSX exports")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_backtest(cfg, args.solax_glob, args.month)


if __name__ == "__main__":
    main()