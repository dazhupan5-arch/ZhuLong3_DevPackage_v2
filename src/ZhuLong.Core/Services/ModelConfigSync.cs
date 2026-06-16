using ZhuLong.Core.Configuration;

namespace ZhuLong.Core.Services;

/// <summary>同步 config 品种列表与安装目录 models/，避免切换 USOIL 时门禁仍只验 XAUUSD。</summary>
public static class ModelConfigSync
{
    private static readonly string[] KnownSymbols = ["XAUUSD", "USOIL"];

    /// <summary>安装目录下含 manifest.json 的品种文件夹。</summary>
    public static IReadOnlyList<string> DiscoverInstalledSymbols()
    {
        var root = AppPaths.ModelsDir;
        if (!Directory.Exists(root))
            return [];

        return Directory.GetDirectories(root)
            .Select(Path.GetFileName)
            .Where(n => !string.IsNullOrWhiteSpace(n)
                        && File.Exists(Path.Combine(root, n!, "manifest.json")))
            .Cast<string>()
            .OrderBy(s => s, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    /// <summary>门禁/UI/历史预热使用的品种并集：default_symbols + 已安装 + 主推理品种。</summary>
    public static IReadOnlyList<string> ResolveSymbols(AppSettings settings)
    {
        var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var s in settings.Model?.DefaultSymbols ?? [])
        {
            if (!string.IsNullOrWhiteSpace(s))
                set.Add(s.Trim());
        }

        foreach (var s in DiscoverInstalledSymbols())
            set.Add(s);

        foreach (var s in KnownSymbols)
        {
            if (Directory.Exists(AppPaths.ModelDir(s)))
                set.Add(s);
        }

        var primary = settings.Model?.PrimarySymbol;
        if (!string.IsNullOrWhiteSpace(primary))
            set.Add(primary.Trim());

        if (set.Count == 0)
            set.Add("XAUUSD");

        return set.OrderBy(s => s, StringComparer.OrdinalIgnoreCase).ToList();
    }

    /// <summary>将缺失的已安装品种写入 default_symbols 并可选保存。</summary>
    public static bool EnsureDefaultSymbols(AppSettings settings, bool persist)
    {
        settings.Model ??= new AppSettings.ModelSettings();
        var target = ResolveSymbols(settings);
        var current = settings.Model.DefaultSymbols ?? [];
        if (current.Length == target.Count
            && current.All(s => target.Contains(s, StringComparer.OrdinalIgnoreCase)))
            return false;

        settings.Model.DefaultSymbols = target.ToArray();
        if (persist)
        {
            try { settings.Save(AppPaths.ConfigPath); }
            catch { /* caller logs */ }
        }

        return true;
    }
}
