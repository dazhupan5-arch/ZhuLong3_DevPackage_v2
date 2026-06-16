using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace ZhuLong.Core.Data;

/// <summary>signals 表列名统一 snake_case（兼容 EF EnsureCreated 产生的 PascalCase 旧库）。</summary>
public static class SignalsSchemaMigrator
{
    private const string StrategyAddSql =
        "ALTER TABLE signals ADD COLUMN strategy TEXT NOT NULL DEFAULT '';";

    public static async Task EnsureReadyAsync(
        ZhuLongDbContext db,
        ILogger? logger = null,
        CancellationToken ct = default)
    {
        await db.Database.ExecuteSqlRawAsync("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id       TEXT PRIMARY KEY,
                timestamp       INTEGER NOT NULL,
                symbol          TEXT NOT NULL,
                direction       TEXT NOT NULL,
                entry_price     REAL NOT NULL,
                stop_loss       REAL NOT NULL,
                take_profit     REAL NOT NULL,
                confidence      REAL NOT NULL,
                expected_return REAL NOT NULL,
                magic_number    INTEGER NOT NULL DEFAULT 0,
                comment_hint    TEXT NOT NULL DEFAULT '',
                status          TEXT NOT NULL DEFAULT 'pending',
                params_snapshot TEXT,
                created_at      INTEGER NOT NULL DEFAULT 0,
                strategy        TEXT NOT NULL DEFAULT ''
            );
            """, ct);

        var cols = await ReadColumnsAsync(db, ct);
        if (cols.Count == 0)
            return;

        if (cols.Contains("signal_id"))
        {
            await EnsureStrategyColumnAsync(db, cols, logger, ct);
            await CreateSignalIndexesAsync(db, ct);
            return;
        }

        if (cols.Contains("SignalId"))
        {
            logger?.LogInformation("迁移 signals: PascalCase → snake_case");
            await MigrateFromPascalCaseAsync(db, ct);
            return;
        }

        logger?.LogWarning("signals 结构异常，列={Columns}，重建空表", string.Join(',', cols));
        await RebuildEmptyAsync(db, ct);
    }

    private static async Task EnsureStrategyColumnAsync(
        ZhuLongDbContext db,
        HashSet<string> cols,
        ILogger? logger,
        CancellationToken ct)
    {
        if (SqliteSchemaHelper.HasColumn(cols, "strategy"))
            return;

        if (SqliteSchemaHelper.HasColumnIgnoreCase(cols, "strategy"))
        {
            logger?.LogInformation("signals 表升级：Strategy → strategy");
        }

        await SqliteSchemaHelper.EnsureSnakeColumnAsync(
            db, "signals", cols, "strategy", "Strategy", StrategyAddSql, ct);
    }

    private static async Task CreateSignalIndexesAsync(ZhuLongDbContext db, CancellationToken ct)
    {
        var cols = await ReadColumnsAsync(db, ct);
        await SqliteSchemaHelper.CreateIndexIfColumnExistsAsync(
            db, cols, "signals", "idx_signals_symbol_ts", "symbol", ct);
        await SqliteSchemaHelper.CreateIndexIfColumnExistsAsync(
            db, cols, "signals", "idx_signals_status", "status", ct);
    }

    private static async Task MigrateFromPascalCaseAsync(ZhuLongDbContext db, CancellationToken ct)
    {
        await SqliteSchemaHelper.TryRenameColumnsAsync(db, "signals",
        [
            ("SignalId", "signal_id"),
            ("Timestamp", "timestamp"),
            ("Symbol", "symbol"),
            ("Direction", "direction"),
            ("EntryPrice", "entry_price"),
            ("StopLoss", "stop_loss"),
            ("TakeProfit", "take_profit"),
            ("Confidence", "confidence"),
            ("ExpectedReturn", "expected_return"),
            ("MagicNumber", "magic_number"),
            ("CommentHint", "comment_hint"),
            ("Status", "status"),
            ("ParamsSnapshot", "params_snapshot"),
            ("CreatedAt", "created_at"),
            ("Strategy", "strategy"),
        ], ct);

        var cols = await ReadColumnsAsync(db, ct);
        if (!cols.Contains("signal_id"))
            await RebuildFromLegacyAsync(db, ct);

        cols = await ReadColumnsAsync(db, ct);
        await SqliteSchemaHelper.EnsureSnakeColumnAsync(
            db, "signals", cols, "strategy", "Strategy", StrategyAddSql, ct);
        await CreateSignalIndexesAsync(db, ct);
    }

    private static async Task RebuildFromLegacyAsync(ZhuLongDbContext db, CancellationToken ct)
    {
        await db.Database.ExecuteSqlRawAsync("""
            CREATE TABLE IF NOT EXISTS signals_new (
                signal_id       TEXT PRIMARY KEY,
                timestamp       INTEGER NOT NULL,
                symbol          TEXT NOT NULL,
                direction       TEXT NOT NULL,
                entry_price     REAL NOT NULL,
                stop_loss       REAL NOT NULL,
                take_profit     REAL NOT NULL,
                confidence      REAL NOT NULL,
                expected_return REAL NOT NULL,
                magic_number    INTEGER NOT NULL DEFAULT 0,
                comment_hint    TEXT NOT NULL DEFAULT '',
                status          TEXT NOT NULL DEFAULT 'pending',
                params_snapshot TEXT,
                created_at      INTEGER NOT NULL DEFAULT 0,
                strategy        TEXT NOT NULL DEFAULT ''
            );
            INSERT INTO signals_new (
                signal_id, timestamp, symbol, direction, entry_price, stop_loss, take_profit,
                confidence, expected_return, magic_number, comment_hint, status, params_snapshot, created_at, strategy
            )
            SELECT SignalId, Timestamp, Symbol, Direction, EntryPrice, StopLoss, TakeProfit,
                   Confidence, ExpectedReturn, MagicNumber, CommentHint, Status, ParamsSnapshot, CreatedAt,
                   COALESCE(Strategy, '')
            FROM signals;
            DROP TABLE signals;
            ALTER TABLE signals_new RENAME TO signals;
            """, ct);
    }

    private static async Task RebuildEmptyAsync(ZhuLongDbContext db, CancellationToken ct)
    {
        await db.Database.ExecuteSqlRawAsync("DROP TABLE IF EXISTS signals;", ct);
        await db.Database.ExecuteSqlRawAsync("""
            CREATE TABLE signals (
                signal_id       TEXT PRIMARY KEY,
                timestamp       INTEGER NOT NULL,
                symbol          TEXT NOT NULL,
                direction       TEXT NOT NULL,
                entry_price     REAL NOT NULL,
                stop_loss       REAL NOT NULL,
                take_profit     REAL NOT NULL,
                confidence      REAL NOT NULL,
                expected_return REAL NOT NULL,
                magic_number    INTEGER NOT NULL DEFAULT 0,
                comment_hint    TEXT NOT NULL DEFAULT '',
                status          TEXT NOT NULL DEFAULT 'pending',
                params_snapshot TEXT,
                created_at      INTEGER NOT NULL DEFAULT 0,
                strategy        TEXT NOT NULL DEFAULT ''
            );
            """, ct);
        await CreateSignalIndexesAsync(db, ct);
    }

    private static Task<HashSet<string>> ReadColumnsAsync(ZhuLongDbContext db, CancellationToken ct) =>
        SqliteSchemaHelper.ReadColumnsAsync(db, "signals", ct);
}
