#!/usr/bin/env python3
"""
build_pr_calendar.py (v2)

Build pr_calendar.json from monthly backtest outputs.

Scans backtest_outputs/YYYY-MM/summary.json and extracts:
- pr_fit_daily_energy  (preferred; produced by backtest_month_v2 / updated backtest_month)
- else falls back to pr_fit (older)

NEW in v2:
- Can emit complete calendar for a given year (YYYY) with all 12 months.
- Fills missing months using a chosen method:
    nearest (default): nearest existing month within the same year
    linear: linear interpolation between nearest left/right months (if both exist)
    mean: yearly mean of existing months
- If a year has no existing months, script will refuse (prints message).

Usage:
  python build_pr_calendar.py
  python build_pr_calendar.py --year 2025
  python build_pr_calendar.py --year 2025 --fill-method linear
  python build_pr_calendar.py --year 2025 --out pr_calendar.json

Notes:
- Output JSON maps "YYYY-MM" -> PR (float).
- Values are clamped to [0.5, 1.0].
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")


def clamp(v: float, lo: float = 0.5, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def parse_month(s: str) -> Tuple[int, int]:
    m = MONTH_RE.match(s.strip())
    if not m:
        raise ValueError("Month must be in YYYY-MM format, e.g. 2025-03")
    return int(m.group(1)), int(m.group(2))


def month_key(y: int, m: int) -> str:
    return f"{y:04d}-{m:02d}"


def month_index(y: int, m: int) -> int:
    return y * 12 + (m - 1)


def extract_pr(summary: dict) -> Optional[float]:
    if "pr_fit_daily_energy" in summary and summary["pr_fit_daily_energy"] is not None:
        try:
            return float(summary["pr_fit_daily_energy"])
        except Exception:
            pass
    if "pr_fit" in summary and summary["pr_fit"] is not None:
        try:
            return float(summary["pr_fit"])
        except Exception:
            pass
    return None


def scan_outputs(outputs_dir: Path) -> Dict[str, float]:
    cal: Dict[str, float] = {}
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs dir not found: {outputs_dir}")

    for p in sorted(outputs_dir.iterdir()):
        if not p.is_dir():
            continue
        key = p.name
        if not MONTH_RE.match(key):
            continue

        summary_path = p / "summary.json"
        if not summary_path.exists():
            continue

        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        pr = extract_pr(summary)
        if pr is None:
            continue

        cal[key] = clamp(pr)

    return cal


def fill_year(cal: Dict[str, float], year: int, method: str) -> Dict[str, float]:
    # existing months for this year
    existing = {}
    for k, v in cal.items():
        y, m = parse_month(k)
        if y == year:
            existing[m] = float(v)

    if not existing:
        raise ValueError(f"No months found in backtest_outputs for year {year}.")

    months = list(range(1, 13))
    out: Dict[str, float] = {}

    if method == "mean":
        mean_val = sum(existing.values()) / len(existing)
        for m in months:
            out[month_key(year, m)] = clamp(existing.get(m, mean_val))
        return out

    def nearest_month(target: int, available: List[int]) -> int:
        return sorted(available, key=lambda a: (abs(a - target), a))[0]

    available = sorted(existing.keys())

    for m in months:
        if m in existing:
            out[month_key(year, m)] = clamp(existing[m])
            continue

        if method == "nearest":
            nm = nearest_month(m, available)
            out[month_key(year, m)] = clamp(existing[nm])
            continue

        if method == "linear":
            left = [a for a in available if a < m]
            right = [a for a in available if a > m]
            if left and right:
                l = max(left)
                r = min(right)
                vl = existing[l]
                vr = existing[r]
                t = (m - l) / (r - l)
                out[month_key(year, m)] = clamp(vl + t * (vr - vl))
            else:
                # fallback to nearest if only one side exists
                nm = nearest_month(m, available)
                out[month_key(year, m)] = clamp(existing[nm])
            continue

        raise ValueError(f"Unknown fill method: {method}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="backtest_outputs", help="Directory containing YYYY-MM folders")
    ap.add_argument("--out", default="pr_calendar.json", help="Output JSON file")
    ap.add_argument("--year", type=int, default=None, help="If set, output will contain ALL 12 months for this year")
    ap.add_argument("--fill-method", default="nearest", choices=["nearest", "linear", "mean"],
                    help="How to fill missing months when --year is used")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs)
    cal = scan_outputs(outputs_dir)

    if not cal:
        print("No PR values found. Did you run backtest_month.py and generate summary.json files?")
        return

    if args.year is not None:
        cal_year = fill_year(cal, args.year, args.fill_method)
        cal_sorted = dict(sorted(cal_year.items(), key=lambda kv: kv[0]))
    else:
        # default: keep only what exists
        cal_sorted = dict(sorted(cal.items(), key=lambda kv: kv[0]))

    out_path = Path(args.out)
    out_path.write_text(json.dumps(cal_sorted, indent=2), encoding="utf-8")

    print(f"Wrote {out_path} with {len(cal_sorted)} months.")
    # print a compact listing
    for k, v in cal_sorted.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()