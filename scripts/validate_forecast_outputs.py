#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


def _nonblank(value: object) -> bool:
    return value is not None and str(value).strip() != ""


def validate_forecast_outputs(summary: Path, intraday_dir: Path, run_date: str) -> None:
    try:
        datetime.strptime(run_date, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(f"Invalid --date {run_date!r}; expected YYYY-MM-DD") from exc

    if not summary.exists():
        raise SystemExit(f"Missing summary CSV: {summary}")

    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    fieldnames = set(rows[0].keys()) if rows else set()
    required_columns = {"Date", "PredictionToday", "PredictionTomorrow"}
    missing_columns = sorted(required_columns - fieldnames)
    if missing_columns:
        raise SystemExit(f"Summary CSV missing required column(s): {', '.join(missing_columns)}")

    date_rows = [row for row in rows if str(row.get("Date", "")) == run_date]
    if not date_rows:
        raise SystemExit(f"Summary CSV has no row for {run_date}")

    row = date_rows[-1]
    missing_values = [
        column
        for column in ("PredictionToday", "PredictionTomorrow")
        if not _nonblank(row.get(column, ""))
    ]
    if missing_values:
        raise SystemExit(
            f"Summary row for {run_date} has blank value(s): {', '.join(missing_values)}"
        )

    intraday = intraday_dir / f"forecast_intraday_{run_date}.csv"
    if not intraday.exists() or intraday.stat().st_size == 0:
        raise SystemExit(f"Missing or empty intraday CSV: {intraday}")

    with intraday.open(newline="", encoding="utf-8") as handle:
        intraday_rows = list(csv.DictReader(handle))

    if not intraday_rows:
        raise SystemExit(f"Intraday CSV has no rows: {intraday}")

    intraday_fieldnames = set(intraday_rows[0].keys())
    if "time" not in intraday_fieldnames:
        raise SystemExit(f"Intraday CSV missing time column: {intraday}")

    day_rows = [row for row in intraday_rows if str(row.get("time", "")).startswith(run_date)]
    if not day_rows:
        raise SystemExit(f"Intraday CSV has no rows for {run_date}: {intraday}")

    value_columns = [column for column in ("pv_kw_pred", "step_kwh") if column in intraday_fieldnames]
    if not value_columns:
        raise SystemExit(f"Intraday CSV missing forecast value columns: {intraday}")

    has_numeric_forecast = False
    for column in value_columns:
        for row in day_rows:
            try:
                float(str(row.get(column, "")).strip())
            except ValueError:
                continue
            has_numeric_forecast = True
            break
        if has_numeric_forecast:
            break
    if not has_numeric_forecast:
        raise SystemExit(f"Intraday CSV has no numeric forecast values for {run_date}: {intraday}")

    print(f"[OK] Forecast outputs validated for {run_date}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Forecast date to validate (YYYY-MM-DD)")
    parser.add_argument("--summary", default="forecasts/forecast_daily_summary.csv")
    parser.add_argument("--intraday-dir", default="forecasts/intraday")
    args = parser.parse_args()

    validate_forecast_outputs(
        summary=Path(args.summary),
        intraday_dir=Path(args.intraday_dir),
        run_date=args.date,
    )


if __name__ == "__main__":
    main()
