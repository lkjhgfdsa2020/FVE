import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests


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
    """
    User azimuth: degrees from north clockwise (0=N, 90=E, 180=S, 270=W).
    Open-Meteo: 0=south, -90=east, +90=west, ±180=north.
    So: openmeteo = az_from_north - 180
    """
    return float(az_from_north) - 180.0


def fetch_open_meteo_forecast(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Open-Meteo allows GTI when tilt + azimuth parameters are set.   [oai_citation:2‡open-meteo.com](https://open-meteo.com/en/docs?utm_source=chatgpt.com)
    panel_az = azimuth_to_openmeteo(cfg.azimuth_deg_from_north)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={cfg.latitude}"
        f"&longitude={cfg.longitude}"
        f"&timezone={cfg.timezone}"
        f"&forecast_days=2"
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

    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(j["daily"]["time"]).dt.date,
            "sunrise": pd.to_datetime(j["daily"]["sunrise"]),
            "sunset": pd.to_datetime(j["daily"]["sunset"]),
        }
    )
    return hourly, daily


def pv_from_gti(cfg: Config, gti_wm2: float) -> float:
    """
    Simple PV model:
      PV_kw = pv_kwp * PR * (GTI / 1000)
    clipped to [0, pv_kwp]
    """
    if gti_wm2 is None or math.isnan(gti_wm2):
        return 0.0
    pv_kw = cfg.pv_kwp * cfg.performance_ratio * (gti_wm2 / 1000.0)
    return float(max(0.0, min(cfg.pv_kwp, pv_kw)))


def propose_off_window(cfg: Config, target: date) -> dict:
    hourly, daily = fetch_open_meteo_forecast(cfg)
    h = hourly[hourly["date"] == target].copy()
    if h.empty:
        return {"error": "Forecast neobsahuje cílový den."}

    drow = daily[daily["date"] == target]
    sunrise = drow.iloc[0]["sunrise"] if not drow.empty else None
    sunset = drow.iloc[0]["sunset"] if not drow.empty else None

    h["pv_kw_pred"] = h["gti"].apply(lambda x: pv_from_gti(cfg, x))

    # Nuluj mimo daylight (pokud sunrise/sunset dostupné)
    if sunrise is not None and sunset is not None:
        h.loc[h["time"] < sunrise, "pv_kw_pred"] = 0.0
        h.loc[h["time"] > sunset, "pv_kw_pred"] = 0.0

    p_thr = cfg.export_limit_kw + cfg.baseload_kw
    exceed = h[h["pv_kw_pred"] > p_thr]
    pv_kwh_pred = float(h["pv_kw_pred"].sum())  # 1h krok => kWh ~ suma kW

    if exceed.empty:
        return {
            "date": str(target),
            "pv_kwh_pred": pv_kwh_pred,
            "t_thr": None,
            "off_start": None,
            "off_end": None,
            "reason": f"PV_pred nikdy nepřekročí práh {p_thr:.2f} kW",
        }

    t_thr = exceed["time"].min()

    # Délka odpojení (heuristika podle síly dne + SoC ráno)
    if pv_kwh_pred >= 25 and cfg.soc_8am >= 70:
        off_hours = 4
    elif pv_kwh_pred >= 20 and cfg.soc_8am >= 50:
        off_hours = 3
    else:
        off_hours = 2
    off_hours = min(off_hours, cfg.max_off_hours)

    off_end = t_thr
    off_start = off_end - pd.Timedelta(hours=off_hours)

    # nezačínej dřív než sunrise (pokud je už po něm)
    if sunrise is not None and off_start < sunrise:
        off_start = sunrise

    return {
        "date": str(target),
        "pv_kwh_pred": pv_kwh_pred,
        "t_thr": t_thr.isoformat(),
        "off_start": off_start.isoformat(),
        "off_end": off_end.isoformat(),
        "reason": f"t_thr=první hodina PV_pred > {p_thr:.2f} kW; off_hours={off_hours}; SoC_8am={cfg.soc_8am:.0f}%",
    }


def main():
    cfg = load_config("config.json")
    today = date.today()
    rec = propose_off_window(cfg, today)
    print(json.dumps(rec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
