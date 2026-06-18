#!/usr/bin/env python3
"""生成归因周报 + 调参建议（读 SQLite 或 JSON 行）。"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.attribution.engine import AttributionEngine
from zhulong.utils.paths import resolve_writable_data_path


def _load_from_sqlite(db_path: Path, limit: int) -> list[dict]:
    if not db_path.is_file():
        return []
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT t.signal_id, t.pnl_percent, t.is_win, t.close_reason,
                   s.attribution_json, s.confidence, s.symbol, s.direction
            FROM trades t
            LEFT JOIN signals s ON s.signal_id = t.signal_id
            WHERE t.close_time IS NOT NULL
            ORDER BY t.close_time DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="", help="trading.db 路径，默认 %APPDATA%/ZhuLong/trading.db")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", default="data/attribution/weekly_report.json")
    parser.add_argument("--min-samples", type=int, default=5)
    args = parser.parse_args()

    db = Path(args.db) if args.db else resolve_writable_data_path("trading.db")
    rows = _load_from_sqlite(db, args.limit)
    engine = AttributionEngine(min_samples=args.min_samples)
    report = engine.analyze(rows)
    out = engine.save_report(report, _ROOT / args.out)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    print(f"saved {out} trades={report.total_trades}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
