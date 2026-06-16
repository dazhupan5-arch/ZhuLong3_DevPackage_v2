using Microsoft.EntityFrameworkCore;

namespace ZhuLong.Core.Data;

internal static class SqliteSchemaHelper
{
    public static async Task<HashSet<string>> ReadColumnsAsync(
        ZhuLongDbContext db,
        string table,
        CancellationToken ct = default)
    {
        var set = new HashSet<string>(StringComparer.Ordinal);
        await using var conn = db.Database.GetDbConnection();
        if (conn.State != System.Data.ConnectionState.Open)
            await conn.OpenAsync(ct);
        await using var cmd = conn.CreateCommand();
        cmd.CommandText = $"PRAGMA table_info({table});";
        await using var reader = await cmd.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
            set.Add(reader.GetString(1));
        return set;
    }

    public static bool HasColumn(HashSet<string> cols, string snakeName) =>
        cols.Contains(snakeName);

    public static bool HasColumnIgnoreCase(HashSet<string> cols, string name) =>
        cols.Any(c => string.Equals(c, name, StringComparison.OrdinalIgnoreCase));

    public static string? FindColumnIgnoreCase(HashSet<string> cols, string name) =>
        cols.FirstOrDefault(c => string.Equals(c, name, StringComparison.OrdinalIgnoreCase));

    public static bool HasPascalColumn(HashSet<string> cols, string pascalName) =>
        cols.Contains(pascalName);

    public static async Task EnsureSnakeColumnAsync(
        ZhuLongDbContext db,
        string table,
        HashSet<string> cols,
        string snakeName,
        string pascalName,
        string addColumnSql,
        CancellationToken ct = default)
    {
        if (HasColumn(cols, snakeName))
            return;

        if (HasColumn(cols, pascalName))
        {
            await TryRenameColumnsAsync(db, table, [(pascalName, snakeName)], ct);
            return;
        }

        var alias = FindColumnIgnoreCase(cols, snakeName);
        if (alias is not null && !string.Equals(alias, snakeName, StringComparison.Ordinal))
        {
            await TryRenameColumnsAsync(db, table, [(alias, snakeName)], ct);
            return;
        }

        if (HasColumnIgnoreCase(cols, snakeName))
            return;

        try
        {
            await db.Database.ExecuteSqlRawAsync(addColumnSql, ct);
        }
        catch
        {
            /* SQLite 列名大小写不敏感，可能已存在 Strategy 等变体 */
        }
    }

    public static async Task TryRenameColumnsAsync(
        ZhuLongDbContext db,
        string table,
        IReadOnlyList<(string From, string To)> renames,
        CancellationToken ct = default)
    {
        foreach (var (from, to) in renames)
        {
            try
            {
                await db.Database.ExecuteSqlRawAsync(
                    $"ALTER TABLE {table} RENAME COLUMN \"{from}\" TO {to};", ct);
            }
            catch
            {
                /* 可能已迁移 */
            }
        }
    }

    public static async Task CreateIndexIfColumnExistsAsync(
        ZhuLongDbContext db,
        HashSet<string> cols,
        string table,
        string indexName,
        string column,
        CancellationToken ct = default)
    {
        if (!HasColumn(cols, column))
            return;

        try
        {
            await db.Database.ExecuteSqlRawAsync(
                $"CREATE INDEX IF NOT EXISTS {indexName} ON {table} ({column});", ct);
        }
        catch
        {
            /* 旧库结构异常时跳过索引 */
        }
    }
}
