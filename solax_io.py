#!/usr/bin/env python3
"""
solax_io.py

Shared helpers for reading SolaXCloud "Inverter Reports" XLSX exports robustly.

SolaXCloud exports sometimes vary in:
- sheet name ("0", "Sheet1", etc.)
- column names (spacing/casing), especially if manually edited

This module tries to locate:
- a timestamp column (Update time)
- a PV power column (Total PV Power (W)) or similar "PV ... Power ... (W)"

It standardizes output to:
- "Update time" (datetime64[ns])
- "Total PV Power (W)" (float)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

TIME_NAME_SET = {"update time", "updatetime", "update_time"}


def _find_time_col(cols) -> Optional[str]:
    for c in cols:
        if str(c).strip().lower() in TIME_NAME_SET:
            return c
    for c in cols:
        s = str(c).lower()
        if "update" in s and "time" in s:
            return c
    return None


def _find_pv_col(cols) -> Optional[str]:
    if "Total PV Power (W)" in cols:
        return "Total PV Power (W)"
    # tolerant: pv + power + (w
    candidates = [c for c in cols if re.search(r"pv.*power", str(c).lower()) and "(w" in str(c).lower()]
    if candidates:
        return candidates[0]
    # further tolerant: any pv + (w
    candidates2 = [c for c in cols if "pv" in str(c).lower() and "(w" in str(c).lower()]
    if candidates2:
        return candidates2[0]
    return None


def read_solax_excel(path: str | Path) -> pd.DataFrame:
    """
    Read a SolaX XLSX export and return normalized columns:
      - Update time
      - Total PV Power (W)
    """
    path = Path(path)

    # Try common sheet name "0" first, else fall back to first sheet.
    try:
        raw = pd.read_excel(path, sheet_name="0")
    except Exception:
        raw = pd.read_excel(path, sheet_name=0)

    # Most SolaX exports have headers in the first row of the sheet (row 0).
    header = raw.iloc[0].tolist()
    df = raw.iloc[1:].copy()
    df.columns = header

    cols = list(df.columns)
    time_col = _find_time_col(cols)
    if time_col is None:
        raise KeyError(
            f"[SolaX] Nenašel jsem časový sloupec (Update time) v souboru: {path}\n"
            f"Dostupné sloupce (ukázka): {cols[:40]}"
        )

    pv_col = _find_pv_col(cols)
    if pv_col is None:
        raise KeyError(
            f"[SolaX] Nenašel jsem PV power sloupec v souboru: {path}\n"
            f"Čekal jsem něco jako 'Total PV Power (W)'.\n"
            f"Dostupné sloupce (ukázka): {cols[:40]}"
        )

    df = df.dropna(subset=[time_col])
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    df[pv_col] = pd.to_numeric(df[pv_col], errors="coerce")
    df = df.dropna(subset=[pv_col])

    return df[[time_col, pv_col]].rename(columns={time_col: "Update time", pv_col: "Total PV Power (W)"})
