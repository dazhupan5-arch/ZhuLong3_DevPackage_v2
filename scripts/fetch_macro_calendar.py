#!/usr/bin/env python3
"""拉取经济日历 CSV 到 data/macro_calendar.csv（离线宏观备用）。"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ZhuLong.PythonEngine"))

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    raise


def fetch_finnhub(token: str, days: int = 7) -> list[dict]:
    url = "https://finnhub.io/api/v1/calendar/economic"
    r = requests.get(url, params={"token": token}, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for e in data.get("economicCalendar", []):
        rows.append({
            "time": e.get("time", ""),
            "country": e.get("country", ""),
            "event": e.get("event", ""),
            "impact": e.get("impact", ""),
            "currency": e.get("currency", ""),
        })
    return rows[: days * 50]


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch macro calendar CSV")
    p.add_argument("--out", default=str(ROOT / "data" / "macro_calendar.csv"))
    p.add_argument("--token", default=os.environ.get("FINNHUB_API_KEY", ""))
    args = p.parse_args()

    if not args.token:
        print("Set FINNHUB_API_KEY or pass --token", file=sys.stderr)
        return 1

    rows = fetch_finnhub(args.token)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time", "country", "event", "impact", "currency"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} events -> {out} @ {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
