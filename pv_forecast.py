import glob
import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import matplotlib.pyplot as plt


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


def azimuth_to_openmeteo(az_from_north: float) -> float:
    # user: 0=N,90=E,180=S,270=W; open-meteo: 0=S, +W / -E
    return float(az_from_north) - 180.0


# ---------- SolaX history -> clear-day envelope ----------
def load_solax_excel(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="0")
    header = raw.iloc[0].tolist()
    df = raw.iloc[1:].copy()
    df.columns = header

    df = df.dropna(subset=["Update time"])
    df["Update time"] = pd.to_datetime(df["Update time"], errors="coerce")
    df = df.dropna(subset=["Update time"])

    df["Total PV Power (W)"] = pd.to_numeric(df["Total PV Power (W)"], errors="coerce")
    df = df.dropna(subset=["Total PV Power (W)"])
    return df


def build_clear_envelope(solax_glob: str = "data/solax/*.xlsx") -> pd.DataFrame:
    paths = sorted(glob.glob(solax_glob))
    if not paths:
        raise FileNotFoundError(
            f"Nenašel jsem žádné SolaX exporty podle '{solax_glob}'. "
            "Dej XLSX exporty do data/solax/."
        )

    df_all = pd.concat([load_solax_excel(p) for p in paths], ignore_index=True)
    df_all = df_all.sort_values("Update time").set_index("Update time")

    # hodinový průměr výkonu (kW)
    pv_kw_h = (df_all["Total PV Power (W)"].resample("h").mean() / 1000.0).rename("pv_kw")

    # denní energie (kWh) = suma hodinových kW
    pv_kwh_d = pv_kw_h.resample("D").sum(min_count=1).rename("pv_kwh")

    daily = pv_kwh_d.reset_index().rename(columns={"Update time": "day"})
    daily["month"] = daily["day"].dt.to_period("M").astype(str)
    daily["date"] = daily["day"].dt.date
    daily["rank"] = daily.groupby("month")["pv_kwh"].rank(pct=True)

    # top 10% dnů (proxy pro jasno)
    clear_days = set(daily.loc[daily["rank"] >= 0.9, "date"])

    hour = pv_kw_h.reset_index().rename(columns={"Update time": "time"})
    hour["date"] = hour["time"].dt.date
    hour["month"] = hour["time"].dt.to_period("M").astype(str)
    hour["hour"] = hour["time"].dt.hour

    # „obálka“: median hodinového výkonu v jasných dnech
    env = (
        hour[hour["date"].isin(clear_days)]
        .groupby(["month", "hour"])["pv_kw"]
        .median()
        .rename("pv_kw_clear_env")
        .reset_index()
    )
    return env


def load_or_build_envelope(cache_path="cache/clear_envelope.parquet", solax_glob="data/solax/*.xlsx") -> pd.DataFrame:
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_parquet(cache)
    env = build_clear_envelope(solax_glob=solax_glob)
    env.to_parquet(cache, index=False)
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


def pv_from_gti(cfg: Config, gti_wm2: float) -> float:
    # PV_kw = kWp * PR * (GTI / 1000), clipped to [0, kWp]
    if gti_wm2 is None or (isinstance(gti_wm2, float) and math.isnan(gti_wm2)):
        return 0.0
    pv_kw = cfg.pv_kwp * cfg.performance_ratio * (float(gti_wm2) / 1000.0)
    return max(0.0, min(cfg.pv_kwp, pv_kw))


def make_hourly_forecast(cfg: Config, env: pd.DataFrame, start_day: date, days: int = 2) -> pd.DataFrame:
    hourly, daily = fetch_open_meteo_forecast(cfg, days=days)

    # přidej sunrise/sunset pro zeroing mimo den
    daily_map = daily.set_index("date")[["sunrise", "sunset"]].to_dict("index")

    h = hourly[hourly["date"].isin([start_day + timedelta(d) for d in range(days)])].copy()
    h["pv_kw_gti"] = h["gti"].apply(lambda x: pv_from_gti(cfg, x))

    # site envelope cap
    h = h.merge(env, on=["month", "hour"], how="left")
    h["pv_kw_clear_env"] = h["pv_kw_clear_env"].fillna(cfg.pv_kwp)  # fallback
    h["pv_kw_pred"] = h[["pv_kw_gti", "pv_kw_clear_env"]].min(axis=1)

    # vynuluj mimo sunrise/sunset
    def in_daylight(row):
        info = daily_map.get(row["date"])
        if not info:
            return True
        return (row["time"] >= info["sunrise"]) and (row["time"] <= info["sunset"])

    mask = h.apply(in_daylight, axis=1)
    h.loc[~mask, "pv_kw_pred"] = 0.0

    # daily kWh
    return h.sort_values("time").reset_index(drop=True)


def plot_today_tomorrow(df: pd.DataFrame, start_day: date):
    days = sorted(df["date"].unique())
    plt.figure(figsize=(11, 5))
    for d in days:
        ddf = df[df["date"] == d].copy()
        # x-axis: hour labels
        x = ddf["time"]
        y = ddf["pv_kw_pred"]
        plt.plot(x, y, label=str(d))

    plt.title("Očekávaný PV výkon (hodinově) – dnes + zítra")
    plt.xlabel("Čas")
    plt.ylabel("Výkon (kW)")
    plt.legend()
    plt.tight_layout()

    out = Path("forecast_plots")
    out.mkdir(exist_ok=True)
    fn = out / f"pv_forecast_{start_day.isoformat()}.png"
    plt.savefig(fn, dpi=160)
    print(f"\nUloženo: {fn}\n")


def main():
    cfg = load_config("config.json")

    # 1) načti / vytvoř envelope z historie (cache)
    env = load_or_build_envelope()

    # 2) udělej forecast na dnes + zítra
    today = date.today()
    df = make_hourly_forecast(cfg, env, today, days=2)

    # 3) vypiš souhrn kWh
    summary = (
        df.groupby("date")["pv_kw_pred"]
        .sum()
        .rename("pv_kwh_pred")
        .reset_index()
    )
    print("\nPredikce denní výroby (kWh):")
    print(summary.to_string(index=False))

    # 4) ulož CSV pro případnou další práci
    out = Path("forecast_outputs")
    out.mkdir(exist_ok=True)
    csv_path = out / f"pv_hourly_forecast_{today.isoformat()}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV uloženo: {csv_path}")

    # 5) vykresli graf
    plot_today_tomorrow(df, today)


if __name__ == "__main__":
    main()
