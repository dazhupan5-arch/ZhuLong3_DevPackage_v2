using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace ZhuLong.Core.Data;

/// <summary>EnsureCreated 不会给旧库补 trades / position_events 表；兼容 PascalCase 旧列名。</summary>
public static class TradesSchemaMigrator
{
    public static async Task EnsureReadyAsync(
        ZhuLongDbContext db,
        ILogger? logger = null,
        CancellationToken ct = default)
    {
        await db.Database.ExecuteSqlRawAsync("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id    TEXT NOT NULL,
                open_time    INTEGER NOT NULL,
                open_price   REAL NOT NULL,
                close_time   INTEGER,
                close_price  REAL,
                pnl_points   REAL,
                pnl_percent  REAL,
                is_win       INTEGER,
                close_reason TEXT
            );
            """, ct);

        await MigrateTradesAsync(db, logger, ct);

        var tradeCols = await SqliteSchemaHelper.ReadColumnsAsync(db, "trades", ct);
        await SqliteSchemaHelper.CreateIndexIfColumnExistsAsync(
            db, tradeCols, "trades", "idx_trades_signal_id", "signal_id", ct);
        await SqliteSchemaHelper.CreateIndexIfColumnExistsAsync(
            db, tradeCols, "trades", "idx_trades_close_time", "close_time", ct);

        await db.Database.ExecuteSqlRawAsync("""
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
            """, ct);

        await MigratePositionEventsAsync(db, logger, ct);

        var eventCols = await SqliteSchemaHelper.ReadColumnsAsync(db, "position_events", ct);
        await SqliteSchemaHelper.CreateIndexIfColumnExistsAsync(
            db, eventCols, "position_events", "idx_position_events_signal_id", "signal_id", ct);

        logger?.LogDebug("trades / position_events 表已就绪");
    }

    private static async Task MigrateTradesAsync(
        ZhuLongDbContext db,
        ILogger? logger,
        CancellationToken ct)
    {
        var cols = await SqliteSchemaHelper.ReadColumnsAsync(db, "trades", ct);
        if (cols.Count == 0 || SqliteSchemaHelper.HasColumn(cols, "signal_id"))
            return;

        if (!SqliteSchemaHelper.HasPascalColumn(cols, "SignalId"))
            return;

        logger?.LogInformation("迁移 trades: PascalCase → snake_case");
        await SqliteSchemaHelper.TryRenameColumnsAsync(db, "trades",
        [
            ("TradeId", "trade_id"),
            ("SignalId", "signal_id"),
            ("OpenTime", "open_time"),
            ("OpenPrice", "open_price"),
            ("CloseTime", "close_time"),
            ("ClosePrice", "close_price"),
            ("PnlPoints", "pnl_points"),
            ("PnlPercent", "pnl_percent"),
            ("IsWin", "is_win"),
            ("CloseReason", "close_reason"),
        ], ct);

        cols = await SqliteSchemaHelper.ReadColumnsAsync(db, "trades", ct);
        if (SqliteSchemaHelper.HasColumn(cols, "signal_id"))
            return;

        await RebuildTradesFromLegacyAsync(db, ct);
    }

    private static async Task RebuildTradesFromLegacyAsync(ZhuLongDbContext db, CancellationToken ct)
    {
        await db.Database.ExecuteSqlRawAsync("""
            CREATE TABLE IF NOT EXISTS trades_new (
                trade_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id    TEXT NOT NULL,
                open_time    INTEGER NOT NULL,
                open_price   REAL NOT NULL,
                close_time   INTEGER,
                close_price  REAL,
                pnl_points   REAL,
                pnl_percent  REAL,
                is_win       INTEGER,
                close_reason TEXT
            );
            INSERT INTO trades_new (
                trade_id, signal_id, open_time, open_price, close_time, close_price,
                pnl_points, pnl_percent, is_win, close_reason
            )
            SELECT TradeId, SignalId, OpenTime, OpenPrice, CloseTime, ClosePrice,
                   PnlPoints, PnlPercent, IsWin, CloseReason
            FROM trades;
            DROP TABLE trades;
            ALTER TABLE trades_new RENAME TO trades;
            """, ct);
    }

    private static async Task MigratePositionEventsAsync(
        ZhuLongDbContext db,
        ILogger? logger,
        CancellationToken ct)
    {
        var cols = await SqliteSchemaHelper.ReadColumnsAsync(db, "position_events", ct);
        if (cols.Count == 0 || SqliteSchemaHelper.HasColumn(cols, "signal_id"))
            return;

        if (!SqliteSchemaHelper.HasPascalColumn(cols, "SignalId"))
            return;

        logger?.LogInformation("迁移 position_events: PascalCase → snake_case");
        await SqliteSchemaHelper.TryRenameColumnsAsync(db, "position_events",
        [
            ("EventId", "event_id"),
            ("SignalId", "signal_id"),
            ("EventTime", "event_time"),
            ("EventType", "event_type"),
            ("Price", "price"),
            ("Volume", "volume"),
            ("OldSl", "old_sl"),
            ("NewSl", "new_sl"),
        ], ct);
    }
}
