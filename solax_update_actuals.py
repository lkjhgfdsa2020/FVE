#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


SUMMARY_CSV_DEFAULT = "forecasts/forecast_daily_summary.csv"


@dataclass
class SolaxConfig:
    base_url: str      # e.g. "https://global.solaxcloud.com"
    token_id: str      # tokenId in headers
    wifi_sn: str       # wifiSn in POST body


def round2(x: float) -> float:
    return float(round(float(x), 2))


def solax_realtime(cfg: SolaxConfig) -> dict[str, Any]:
    """
    POST {base_url}/api/v2/dataAccess/realtimeInfo/get
    Headers: tokenId: <tokenId>
    Body: {"wifiSn":"..."}
    Returns 'result' with fields including yieldtoday, yieldtotal, uploadTime, etc.
    """
    url = f"{cfg.base_url.rstrip('/')}/api/v2/dataAccess/realtimeInfo/get"
    headers = {"tokenId": cfg.token_id, "Content-Type": "application/json"}
    payload = {"wifiSn": cfg.wifi_sn}

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    j = r.json()

    if not j.get("success", False):
        raise RuntimeError(f"SolaX API error: success=false, exception={j.get('exception')}, code={j.get('code')}")
    if "result" not in j or not isinstance(j["result"], dict):
        raise RuntimeError("SolaX API response missing 'result' object.")
    return j


def ensure_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Date", "PredictionToday", "ActualToday", "PredictionTomorrow", "SwitchOff", "SwitchOn"]
    for c in required:
        if c not in df.columns:
            df[c] = pd.NA
    df["Date"] = df["Date"].astype(str)
    return df[required]


def upsert_actual_today(summary_csv: str, day: date, actual_kwh: float) -> None:
    p = Path(summary_csv)
    if not p.exists():
        raise FileNotFoundError(f"Missing summary csv: {p}")

    df = pd.read_csv(p)
    df = ensure_summary_columns(df)

    ds = day.isoformat()
    if (df["Date"] == ds).any():
        df.loc[df["Date"] == ds, "ActualToday"] = round2(actual_kwh)
    else:
        # create row if missing (preds/switches may arrive later)
        df = pd.concat(
            [df, pd.DataFrame([{
                "Date": ds,
                "PredictionToday": pd.NA,
                "ActualToday": round2(actual_kwh),
                "PredictionTomorrow": pd.NA,
                "SwitchOff": "",
                "SwitchOn": "",
            }])],
            ignore_index=True,
        )

    df = df.sort_values("Date")
    df.to_csv(p, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=SUMMARY_CSV_DEFAULT)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--token-id", required=True)
    ap.add_argument("--wifi-sn", required=True)
    ap.add_argument("--date", default=None, help="Override 'today' (YYYY-MM-DD) for testing")
    args = ap.parse_args()

    today = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()

    cfg = SolaxConfig(
        base_url=args.base_url,
        token_id=args.token_id,
        wifi_sn=args.wifi_sn,
    )

    j = solax_realtime(cfg)
    result = j["result"]

    if "yieldtoday" not in result:
        raise RuntimeError("SolaX result missing 'yieldtoday'.")

    actual = float(result["yieldtoday"])
    upload_time = result.get("uploadTime", "")

    upsert_actual_today(args.summary, today, actual)
    print(f"[OK] ActualToday updated for {today.isoformat()}: {round2(actual)} kWh (uploadTime={upload_time})")


if __name__ == "__main__":
    main()
