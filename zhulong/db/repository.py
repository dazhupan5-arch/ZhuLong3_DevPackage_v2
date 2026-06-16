"""SQLite 数据库初始化与访问。"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from zhulong.utils.paths import database_path

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS signals (
    signal_id       TEXT PRIMARY KEY,
    timestamp       INTEGER NOT NULL,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('buy', 'sell')),
    entry_price     REAL NOT NULL,
    stop_loss       REAL NOT NULL,
    take_profit     REAL NOT NULL,
    confidence      REAL NOT NULL,
    expected_return REAL NOT NULL,
    magic_number    INTEGER NOT NULL,
    comment_hint    TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'matched', 'expired', 'rejected')),
    params_snapshot TEXT,
    created_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id    TEXT NOT NULL REFERENCES signals (signal_id),
    open_time    INTEGER NOT NULL,
    open_price   REAL NOT NULL,
    close_time   INTEGER,
    close_price  REAL,
    pnl_points   REAL,
    pnl_percent  REAL,
    is_win       INTEGER CHECK (is_win IN (0, 1)),
    close_reason TEXT
);

CREATE TABLE IF NOT EXISTS position_events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id  TEXT NOT NULL,
    event_time INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    price      REAL,
    volume     REAL,
    old_sl     REAL,
    new_sl     REAL
);
"""


def get_connection() -> sqlite3.Connection:
    db_path = database_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    conn = get_connection()
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info("数据库就绪: %s", database_path())
    finally:
        conn.close()


def insert_signal(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO signals (
            signal_id, timestamp, symbol, direction, entry_price, stop_loss,
            take_profit, confidence, expected_return, magic_number, comment_hint,
            status, params_snapshot, created_at
        ) VALUES (
            :signal_id, :timestamp, :symbol, :direction, :entry_price, :stop_loss,
            :take_profit, :confidence, :expected_return, :magic_number, :comment_hint,
            :status, :params_snapshot, :created_at
        )
        """,
        row,
    )
    conn.commit()
