using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using ZhuLong.Core.Data;

namespace ZhuLong.Core.Tests;

/// <summary>本地诊断：ZHULONG_MIGRATE_DB=path dotnet test --filter MigrateLiveDatabase</summary>
public class LiveDatabaseMigrationTests
{
    [Fact]
    public async Task MigrateLiveDatabase_WhenEnvSet()
    {
        var path = Environment.GetEnvironmentVariable("ZHULONG_MIGRATE_DB");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return;

        var options = new DbContextOptionsBuilder<ZhuLongDbContext>()
            .UseSqlite($"Data Source={path}")
            .Options;

        await using var db = new ZhuLongDbContext(options);
        await SignalsSchemaMigrator.EnsureReadyAsync(db);
        await TradesSchemaMigrator.EnsureReadyAsync(db);

        var signalCols = await SqliteSchemaHelper.ReadColumnsAsync(db, "signals");
        Assert.Contains("signal_id", signalCols);

        var tradeCols = await SqliteSchemaHelper.ReadColumnsAsync(db, "trades");
        if (tradeCols.Count > 0)
            Assert.Contains("signal_id", tradeCols);

        await db.Database.CloseConnectionAsync();
        SqliteConnection.ClearAllPools();
    }
}
