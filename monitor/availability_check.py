#!/usr/bin/env python3
"""Availability monitor for inverter connection/control percentages (E.ON DeltaGreen).

Auth model observed from browser request:
- Cookie-based session: proteus_session=...
- CSRF cookie: proteus_csrf=...
- CSRF header required: x-proteus-csrf: <same value as proteus_csrf>

This script:
- Fetches current-month record (Europe/Prague) and reads:
    * dataPointPercentage -> DisponibilitaPripojeni (0..1)
    * controlPercentage   -> DisponibilitaRizeni   (0..1)
- Notifies via email (Gmail app password) when level changes:
    OK <-> WARN (<92%) <-> CRIT (<90%)

Secrets/env expected (GitHub Actions):
- AVAILABILITY_API_URL
- AVAILABILITY_COOKIE
- AVAILABILITY_CSRF
- AVAILABILITY_REFERER (recommended)
- AVAILABILITY_USER_AGENT (optional)

- SMTP_USER, SMTP_PASSWORD, MAIL_TO, MAIL_FROM (optional)
"""

from __future__ import annotations

import json
import os
import smtplib
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from typing import Any, Dict, Optional, Tuple

import requests
from zoneinfo import ZoneInfo

STATE_PATH = os.environ.get("STATE_PATH", "monitor/state.json")

WARN_THRESHOLD_PCT = float(os.environ.get("WARN_THRESHOLD_PCT", "92"))
CRIT_THRESHOLD_PCT = float(os.environ.get("CRIT_THRESHOLD_PCT", "90"))
WARN_THRESHOLD = WARN_THRESHOLD_PCT / 100.0
CRIT_THRESHOLD = CRIT_THRESHOLD_PCT / 100.0

PRAGUE_TZ = ZoneInfo(os.environ.get("LOCAL_TZ", "Europe/Prague"))


@dataclass(frozen=True)
class AvailabilitySample:
    dispon_pripojeni: float
    dispon_rizeni: float
    month_start_utc: datetime
    month_start_local: datetime


def _load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_level": "OK", "last_notified_at": None, "last_month": None}


def _save_state(state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _iso_z_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _pick_current_month_record(records: list[dict]) -> dict:
    now_local = datetime.now(tz=PRAGUE_TZ)
    target_year, target_month = now_local.year, now_local.month

    best: Optional[dict] = None
    for r in records:
        m = r.get("month")
        if not m:
            continue
        dt_utc = _iso_z_to_dt(m)
        dt_local = dt_utc.astimezone(PRAGUE_TZ)
        if (dt_local.year, dt_local.month) == (target_year, target_month):
            if best is None:
                best = r
            else:
                prev_dt = _iso_z_to_dt(best["month"])
                if dt_utc > prev_dt:
                    best = r

    if best is None:
        latest: Optional[Tuple[datetime, dict]] = None
        for r in records:
            m = r.get("month")
            if not m:
                continue
            dt_utc = _iso_z_to_dt(m)
            dt_local = dt_utc.astimezone(PRAGUE_TZ)
            if (dt_local.year, dt_local.month) <= (target_year, target_month):
                if latest is None or dt_utc > latest[0]:
                    latest = (dt_utc, r)
        if latest is None:
            raise ValueError("No usable 'records' entries with a 'month' field.")
        best = latest[1]

    return best


def _sanitize_single_line(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v2 = v.replace("\r", "").replace("\n", "").strip()
    if not v2 or v2 == "***":
        return None
    return v2


def _build_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}

    cookie = _sanitize_single_line(os.environ.get("AVAILABILITY_COOKIE"))
    if cookie:
        headers["cookie"] = cookie

    csrf = _sanitize_single_line(os.environ.get("AVAILABILITY_CSRF"))
    if csrf:
        headers["x-proteus-csrf"] = csrf

    referer = _sanitize_single_line(os.environ.get("AVAILABILITY_REFERER"))
    if referer:
        headers["referer"] = referer
        headers["origin"] = "https://eon.deltagreen.cz"

    headers["accept"] = "application/json"
    headers["content-type"] = "application/json"
    headers["trpc-accept"] = "application/jsonl"

    ua = os.environ.get(
        "AVAILABILITY_USER_AGENT",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    )
    headers["user-agent"] = _sanitize_single_line(ua) or ua

    return headers


def fetch_current_month_availability() -> AvailabilitySample:
    url = os.environ["AVAILABILITY_API_URL"].strip()
    headers = _build_headers()

    r = requests.get(url, headers=headers, timeout=30)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        snippet = (r.text or "")[:500]
        print(f"HTTP {r.status_code} from API. Body (first 500 chars):\n{snippet}", file=sys.stderr)
        raise

    data = r.json()

    def find_records(obj: Any) -> Optional[list[dict]]:
        if isinstance(obj, dict):
            if "records" in obj and isinstance(obj["records"], list):
                return obj["records"]
            for v in obj.values():
                rec = find_records(v)
                if rec is not None:
                    return rec
        elif isinstance(obj, list):
            for it in obj:
                rec = find_records(it)
                if rec is not None:
                    return rec
        return None

    records = find_records(data)
    if not records:
        raise ValueError("Could not find 'records' list in API response.")

    rec = _pick_current_month_record(records)

    dp = float(rec["dataPointPercentage"])
    cp = float(rec["controlPercentage"])
    month_utc = _iso_z_to_dt(rec["month"])
    month_local = month_utc.astimezone(PRAGUE_TZ)

    return AvailabilitySample(dp, cp, month_utc, month_local)


def classify(sample: AvailabilitySample) -> str:
    worst = min(sample.dispon_pripojeni, sample.dispon_rizeni)
    if worst < CRIT_THRESHOLD:
        return "CRIT"
    if worst < WARN_THRESHOLD:
        return "WARN"
    return "OK"


def _send_email(subject: str, body: str) -> None:
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))

    smtp_user = os.environ["SMTP_USER"].strip()
    smtp_password = _sanitize_single_line(os.environ.get("SMTP_PASSWORD")) or ""
    if not smtp_password:
        raise RuntimeError("SMTP_PASSWORD is empty after sanitization (check secret formatting).")

    mail_from = os.environ.get("MAIL_FROM", smtp_user).strip()
    mail_to = os.environ["MAIL_TO"].strip()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def main() -> int:
    state = _load_state()
    prev_level = state.get("last_level", "OK")

    sample = fetch_current_month_availability()
    level = classify(sample)

    subject = f"FVE Availability {level} - {sample.month_start_local.strftime('%Y-%m')}"
    body = (
        f"FVE Availability monitor status: {level}\n"
        f"Month (local): {sample.month_start_local.strftime('%Y-%m')} (Europe/Prague)\n"
        f"DisponibilitaPripojeni (dataPointPercentage): {sample.dispon_pripojeni*100:.2f}%\n"
        f"DisponibilitaRizeni (controlPercentage): {sample.dispon_rizeni*100:.2f}%\n"
        f"Previous level: {prev_level}\n"
        f"Thresholds: WARN<{WARN_THRESHOLD_PCT:.0f}%, CRIT<{CRIT_THRESHOLD_PCT:.0f}%\n"
        f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
    )

    if level != prev_level:
        _send_email(subject, body)
        state["last_level"] = level
        state["last_notified_at"] = int(time.time())
        state["last_month"] = sample.month_start_local.strftime("%Y-%m")
        _save_state(state)
    else:
        state["last_month"] = sample.month_start_local.strftime("%Y-%m")
        _save_state(state)

    return 2 if level in ("WARN", "CRIT") else 0


if __name__ == "__main__":
    sys.exit(main())
