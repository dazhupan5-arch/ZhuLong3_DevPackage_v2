using ZhuLong.Core.Configuration;
using ZhuLong.Core.Macro;
using ZhuLong.Core.Services;

namespace ZhuLong.Core.Bootstrap;

/// <summary>首次启动：创建 logs/、AppData config、目录结构。</summary>
public static class AppBootstrap
{
    public static AppSettings EnsureFirstRun()
    {
        _ = AppPaths.LogsDir;
        _ = AppPaths.AppDataDir;
        _ = AppPaths.WritableDataDir;
        MacroBundledDataSync.SeedOptionalJsonIfMissing();
        MacroBundledDataSync.SyncMacroEventsCsvFromInstall();

        var userConfig = Path.Combine(AppPaths.AppDataDir, "config.json");
        var installConfig = Path.Combine(AppPaths.InstallDir, "config.json");

        AgentConfigSync.ForceCopyMainConfigIfInstallUpgraded();

        if (!File.Exists(userConfig) || new FileInfo(userConfig).Length < 64)
        {
            if (File.Exists(installConfig))
                File.Copy(installConfig, userConfig, overwrite: true);
            else
                new AppSettings().Save(userConfig);
        }
        else
        {
            MergeMissingFromInstall(userConfig, installConfig);
        }

        var schemaErrors = ConfigSchemaValidator.ValidateFile(userConfig);
        if (schemaErrors.Count > 0)
        {
            var log = Path.Combine(AppPaths.LogsDir, "config_schema_errors.txt");
            File.WriteAllLines(log, schemaErrors);
        }

        var settings = AppSettings.LoadOrCreate(userConfig);
        AgentConfigSync.EnsureUserAgentConfig(settings);
        AgentConfigSync.AlignWithAppSettings(settings);
        SeedBundledAgentFiles();
        return settings;
    }

    private static void SeedBundledAgentFiles()
    {
        foreach (var rel in new[]
        {
            "data/agent_state_scaler_xauusd.json",
            "data/agent_state_scaler_usoil.json",
        })
        {
            var dst = Path.Combine(AppPaths.WritableDataDir, rel.Replace("data/", "", StringComparison.Ordinal));
            if (File.Exists(dst))
                continue;
            var src = Path.Combine(AppPaths.InstallDir, rel.Replace('/', Path.DirectorySeparatorChar));
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
    }

    /// <summary>升级安装后补齐用户 config 中缺失的块（如 risk_guard）。</summary>
    private static void MergeMissingFromInstall(string userConfig, string installConfig)
    {
        if (!File.Exists(installConfig)) return;

        var user = AppSettings.Load(userConfig);
        var install = AppSettings.Load(installConfig);
        var changed = false;

        if (user.RiskGuard is null && install.RiskGuard is not null)
        {
            user.RiskGuard = install.RiskGuard;
            changed = true;
        }

        if (user.TradingAgent is null && install.TradingAgent is not null)
        {
            user.TradingAgent = install.TradingAgent;
            changed = true;
        }

        if (user.MultiStrategy is null && install.MultiStrategy is not null)
        {
            user.MultiStrategy = install.MultiStrategy;
            changed = true;
        }

        if (ModelConfigSync.EnsureDefaultSymbols(user, persist: false))
            changed = true;
        else
        {
            var installSymbols = install.Model?.DefaultSymbols;
            if (installSymbols is { Length: > 0 })
            {
                var merged = ModelConfigSync.ResolveSymbols(user).ToList();
                foreach (var s in installSymbols)
                {
                    if (!merged.Contains(s, StringComparer.OrdinalIgnoreCase))
                        merged.Add(s);
                }

                var cur = user.Model?.DefaultSymbols ?? [];
                if (merged.Count != cur.Length || merged.Any(s => !cur.Contains(s, StringComparer.OrdinalIgnoreCase)))
                {
                    user.Model ??= new AppSettings.ModelSettings();
                    user.Model.DefaultSymbols = merged.OrderBy(s => s, StringComparer.OrdinalIgnoreCase).ToArray();
                    changed = true;
                }
            }
        }

        if (changed)
            user.Save(userConfig);
    }
}
