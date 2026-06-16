using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace ZhuLong.Core.Data;

/// <summary>macro_events 表列名统一为 snake_case（兼容旧 PascalCase 库）。</summary>
public static class MacroEventsSchemaMigrator
{
    /// <summary>EnsureCreated 不会给已有库补表；显式 CREATE IF NOT EXISTS。</summary>
    public static async Task EnsureTableAsync(ZhuLongDbContext db, CancellationToken ct = default)
    {
        await db.Database.ExecuteSqlRawAsync("""
            CREATE TABLE IF NOT EXISTS macro_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time_unix INTEGER NOT NULL,
                event_name TEXT NOT NULL,
                impact TEXT NOT NULL,
                currency TEXT NOT NULL,
                source TEXT NOT NULL,
                fetched_at_unix INTEGER NOT NULL,
                external_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_macro_events_time ON macro_events(event_time_unix);
            """, ct);
    }

    public static async Task EnsureReadyAsync(
        ZhuLongDbContext db,
        ILogger? logger = null,
        CancellationToken ct = default)
    {
        await EnsureTableAsync(db, ct);
        await EnsureSnakeCaseAsync(db, logger, ct);
    }

    public static async Task EnsureSnakeCaseAsync(
        ZhuLongDbContext db,
        ILogger? logger = null,
        CancellationToken ct = default)
    {
        var cols = await ReadColumnsAsync(db, ct);
        if (cols.Count == 0)
            return;

        if (cols.Contains("event_time_unix"))
            return;

        if (!cols.Contains("EventTimeUnix"))
        {
            logger?.LogWarning("macro_events 结构异常，列={Columns}", string.Join(',', cols));
            await RebuildEmptyAsync(db, ct);
            return;
        }

        logger?.LogInformation("迁移 macro_events: PascalCase → snake_case");
        if (await TryRenameAsync(db, ct))
            return;

        await RebuildFromLegacyAsync(db, ct);
    }

    private static async Task<List<string>> ReadColumnsAsync(ZhuLongDbContext db, CancellationToken ct)
    {
        var conn = db.Database.GetDbConnection();
        if (conn.State != System.Data.ConnectionState.Open)
            await conn.OpenAsync(ct);

        var cols = new List<string>();
        await using var cmd = conn.CreateCommand();
        cmd.CommandText = "SELECT name FROM pragma_table_info('macro_events')";
        await using var reader = await cmd.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
            cols.Add(reader.GetString(0));
        return cols;
    }

    private static async Task<bool> TryRenameAsync(ZhuLongDbContext db, CancellationToken ct)
    {
        string[] renames =
        [
            "ALTER TABLE macro_events RENAME COLUMN \"Id\" TO id",
            "ALTER TABLE macro_events RENAME COLUMN \"EventTimeUnix\" TO event_time_unix",
            "ALTER TABLE macro_events RENAME COLUMN \"EventName\" TO event_name",
            "ALTER TABLE macro_events RENAME COLUMN \"Impact\" TO impact",
            "ALTER TABLE macro_events RENAME COLUMN \"Currency\" TO currency",
            "ALTER TABLE macro_events RENAME COLUMN \"Source\" TO source",
            "ALTER TABLE macro_events RENAME COLUMN \"FetchedAtUnix\" TO fetched_at_unix",
            "ALTER TABLE macro_events RENAME COLUMN \"ExternalId\" TO external_id",
        ];

        foreach (var sql in renames)
        {
            try { await db.Database.ExecuteSqlRawAsync(sql, ct); }
            catch { /* 可能已迁移 */ }
        }

        var cols = await ReadColumnsAsync(db, ct);
        return cols.Contains("event_time_unix");
    }

    private static async Task RebuildFromLegacyAsync(ZhuLongDbContext db, CancellationToken ct)
    {
        await db.Database.ExecuteSqlRawAsync("""
            CREATE TABLE IF NOT EXISTS macro_events_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time_unix INTEGER NOT NULL,
                event_name TEXT NOT NULL,
                impact TEXT NOT NULL,
                currency TEXT NOT NULL,
                source TEXT NOT NULL,
                fetched_at_unix INTEGER NOT NULL,
                external_id TEXT
            );
            INSERT INTO macro_events_new (event_time_unix, event_name, impact, currency, source, fetched_at_unix, external_id)
            SELECT EventTimeUnix, EventName, Impact, Currency, Source, FetchedAtUnix, ExternalId FROM macro_events;
            DROP TABLE macro_events;
            ALTER TABLE macro_events_new RENAME TO macro_events;
            CREATE INDEX IF NOT EXISTS idx_macro_events_time ON macro_events(event_time_unix);
            """, ct);
    }

    private static async Task RebuildEmptyAsync(ZhuLongDbContext db, CancellationToken ct)
    {
        await db.Database.ExecuteSqlRawAsync("DROP TABLE IF EXISTS macro_events;", ct);
        await db.Database.ExecuteSqlRawAsync("""
            CREATE TABLE macro_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time_unix INTEGER NOT NULL,
                event_name TEXT NOT NULL,
                impact TEXT NOT NULL,
                currency TEXT NOT NULL,
                source TEXT NOT NULL,
                fetched_at_unix INTEGER NOT NULL,
                external_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_macro_events_time ON macro_events(event_time_unix);
            """, ct);
    }
}
