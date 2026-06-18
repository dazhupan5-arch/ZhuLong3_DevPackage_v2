using Microsoft.Extensions.Logging;

namespace ZhuLong.Core.Macro;

/// <summary>安装包 data/ 与 AppData 可写目录同步（升级后同事机也能拿到最新宏观 CSV）。</summary>
public static class MacroBundledDataSync
{
    /// <summary>
    /// 每次启动用安装目录的 macro_events.csv 覆盖 AppData 副本（安装包为 Tier1 事件权威来源）。
    /// </summary>
    public static bool SyncMacroEventsCsvFromInstall(ILogger? logger = null)
    {
        var src = Path.Combine(AppPaths.InstallDir, "data", "macro_events.csv");
        var dst = AppPaths.MacroEventsPath;
        if (!File.Exists(src))
            return false;

        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
            var changed = !File.Exists(dst) || !FilesEqual(src, dst);
            File.Copy(src, dst, overwrite: true);
            if (changed)
                logger?.LogInformation("宏观日历 CSV 已从安装包同步（含 FOMC 等 Tier1 北京时间） path={Path}", dst);
            return changed;
        }
        catch (Exception ex)
        {
            logger?.LogWarning(ex, "同步 macro_events.csv 失败 src={Src}", src);
            return false;
        }
    }

    /// <summary>首次安装时复制 fred/sentiment；macro_events 始终走 SyncMacroEventsCsvFromInstall。</summary>
    public static void SeedOptionalJsonIfMissing()
    {
        foreach (var name in new[] { "fred_latest.json", "sentiment.json" })
        {
            var dst = Path.Combine(AppPaths.WritableDataDir, name);
            if (File.Exists(dst))
                continue;
            var src = Path.Combine(AppPaths.InstallDir, "data", name);
            if (!File.Exists(src))
                continue;
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
                File.Copy(src, dst, overwrite: false);
            }
            catch
            {
                /* ignore */
            }
        }

        var bundledMacro = Path.Combine(AppPaths.InstallDir, "data", "macro");
        var userMacro = Path.Combine(AppPaths.WritableDataDir, "macro");
        var bundledCsv = Path.Combine(bundledMacro, "macro_daily.csv");
        var userCsv = Path.Combine(userMacro, "macro_daily.csv");
        if (File.Exists(bundledCsv) && !File.Exists(userCsv))
        {
            try
            {
                Directory.CreateDirectory(userMacro);
                File.Copy(bundledCsv, userCsv, overwrite: false);
            }
            catch
            {
                /* ignore */
            }
        }
    }

    private static bool FilesEqual(string a, string b)
    {
        try
        {
            var fa = new FileInfo(a);
            var fb = new FileInfo(b);
            if (fa.Length != fb.Length)
                return false;
            return string.Equals(
                File.ReadAllText(a),
                File.ReadAllText(b),
                StringComparison.Ordinal);
        }
        catch
        {
            return false;
        }
    }
}
