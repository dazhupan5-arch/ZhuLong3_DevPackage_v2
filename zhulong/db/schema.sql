-- 烛龙三代 trading.db · 见 docs/DECISIONS.md G9
-- 执行: sqlite3 data/trading.db < zhulong/db/schema.sql

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

CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals (symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status);

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
    close_reason TEXT CHECK (close_reason IN (
        'tp', 'sl', 'time_stop', 'trailing', 'model_exit', 'manual'
    ))
);

CREATE INDEX IF NOT EXISTS idx_trades_signal_id ON trades (signal_id);
CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trades (close_time);

CREATE TABLE IF NOT EXISTS position_events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id  TEXT NOT NULL,
    event_time INTEGER NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN (
        'partial_close', 'move_sl', 'move_tp', 'full_close'
    )),
    price      REAL,
    volume     REAL,
    old_sl     REAL,
    new_sl     REAL
);

CREATE INDEX IF NOT EXISTS idx_position_events_signal_id ON position_events (signal_id);
