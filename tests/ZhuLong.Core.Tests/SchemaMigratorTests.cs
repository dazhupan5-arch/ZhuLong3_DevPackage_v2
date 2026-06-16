using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using ZhuLong.Core.Data;

namespace ZhuLong.Core.Tests;

public class SchemaMigratorTests
{
    private static async Task DisposeDbAsync(string path)
    {
        SqliteConnection.ClearAllPools();
        for (var i = 0; i < 5; i++)
        {
            try
            {
                if (File.Exists(path))
                    File.Delete(path);
                return;
            }
            catch (IOException) when (i < 4)
            {
                await Task.Delay(50);
            }
        }
    }

    [Fact]
    public async Task TradesMigrator_RenamesPascalCase_AndCreatesIndexes()
    {
        var path = Path.Combine(Path.GetTempPath(), $"zhulong_trades_test_{Guid.NewGuid():N}.db");
        try
        {
            var options = new DbContextOptionsBuilder<ZhuLongDbContext>()
                .UseSqlite($"Data Source={path}")
                .Options;

            await using (var seed = new ZhuLongDbContext(options))
            {
                await seed.Database.ExecuteSqlRawAsync("""
                    CREATE TABLE trades (
                        TradeId     INTEGER PRIMARY KEY AUTOINCREMENT,
                        SignalId    TEXT NOT NULL,
                        OpenTime    INTEGER NOT NULL,
                        OpenPrice   REAL NOT NULL,
                        CloseTime   INTEGER,
                        ClosePrice  REAL,
                        PnlPoints   REAL,
                        PnlPercent  REAL,
                        IsWin       INTEGER,
                        CloseReason TEXT
                    );
                    """);
            }

            await using var db = new ZhuLongDbContext(options);
            await TradesSchemaMigrator.EnsureReadyAsync(db);

            var cols = await SqliteSchemaHelper.ReadColumnsAsync(db, "trades");
            Assert.Contains("signal_id", cols);
            Assert.Contains("trade_id", cols);

            var trades = await db.Trades.ToListAsync();
            Assert.Empty(trades);
        }
        finally
        {
            await DisposeDbAsync(path);
        }
    }

    [Fact]
    public async Task SignalsMigrator_RenamesPascalCase()
    {
        var path = Path.Combine(Path.GetTempPath(), $"zhulong_signals_test_{Guid.NewGuid():N}.db");
        try
        {
            var options = new DbContextOptionsBuilder<ZhuLongDbContext>()
                .UseSqlite($"Data Source={path}")
                .Options;

            await using (var seed = new ZhuLongDbContext(options))
            {
                await seed.Database.ExecuteSqlRawAsync("""
                    CREATE TABLE signals (
                        SignalId       TEXT PRIMARY KEY,
                        Timestamp      INTEGER NOT NULL,
                        Symbol         TEXT NOT NULL,
                        Direction      TEXT NOT NULL,
                        EntryPrice     REAL NOT NULL,
                        StopLoss       REAL NOT NULL,
                        TakeProfit     REAL NOT NULL,
                        Confidence     REAL NOT NULL,
                        ExpectedReturn REAL NOT NULL,
                        MagicNumber    INTEGER NOT NULL DEFAULT 0,
                        CommentHint    TEXT NOT NULL DEFAULT '',
                        Status         TEXT NOT NULL DEFAULT 'pending',
                        ParamsSnapshot TEXT,
                        CreatedAt      INTEGER NOT NULL DEFAULT 0
                    );
                    """);
            }

            await using var db = new ZhuLongDbContext(options);
            await SignalsSchemaMigrator.EnsureReadyAsync(db);

            var cols = await SqliteSchemaHelper.ReadColumnsAsync(db, "signals");
            Assert.Contains("signal_id", cols);
            Assert.Contains("strategy", cols);

            var count = await db.Signals.CountAsync();
            Assert.Equal(0, count);
        }
        finally
        {
            await DisposeDbAsync(path);
        }
    }

    [Fact]
    public async Task SignalsMigrator_RenamesPascalStrategy_WhenOtherColumnsSnakeCase()
    {
        var path = Path.Combine(Path.GetTempPath(), $"zhulong_signals_mixed_{Guid.NewGuid():N}.db");
        try
        {
            var options = new DbContextOptionsBuilder<ZhuLongDbContext>()
                .UseSqlite($"Data Source={path}")
                .Options;

            await using (var seed = new ZhuLongDbContext(options))
            {
                await seed.Database.ExecuteSqlRawAsync("""
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
                        Strategy        TEXT NOT NULL DEFAULT ''
                    );
                    """);
            }

            await using var db = new ZhuLongDbContext(options);
            await SignalsSchemaMigrator.EnsureReadyAsync(db);
            await SignalsSchemaMigrator.EnsureReadyAsync(db);

            var cols = await SqliteSchemaHelper.ReadColumnsAsync(db, "signals");
            Assert.Contains("strategy", cols);
            Assert.DoesNotContain("Strategy", cols);

            var count = await db.Signals.CountAsync();
            Assert.Equal(0, count);
        }
        finally
        {
            await DisposeDbAsync(path);
        }
    }
}
