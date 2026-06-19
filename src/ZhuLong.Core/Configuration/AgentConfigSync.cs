using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.Logging;

namespace ZhuLong.Core.Configuration;

/// <summary>保持 config.json trading_agent.enabled 与 config_agent.json enabled 一致。</summary>
public static class AgentConfigSync
{
    /// <summary>用户可写的 config_agent.json（Program Files 安装目录常只读）。</summary>
    public static string UserAgentConfigPath =>
        Path.Combine(AppPaths.AppDataDir, "config_agent.json");

    public static (bool Ok, bool Changed) TrySyncEnabled(string agentConfigPath, bool enabled, ILogger? logger = null)
    {
        if (string.IsNullOrWhiteSpace(agentConfigPath) || !File.Exists(agentConfigPath))
            return (false, false);

        try
        {
            var text = File.ReadAllText(agentConfigPath);
            if (text.Length > 0 && text[0] == '\uFEFF')
                text = text[1..];

            var node = JsonNode.Parse(text);
            if (node is not JsonObject root)
                return (false, false);

            if (root.TryGetPropertyValue("enabled", out var cur) && cur is JsonValue jv &&
                jv.TryGetValue<bool>(out var curEnabled) && curEnabled == enabled)
                return (true, false);

            root["enabled"] = enabled;
            var outText = root.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(agentConfigPath, outText, new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            logger?.LogInformation("已同步 config_agent.json enabled={Enabled} path={Path}", enabled, agentConfigPath);
            return (true, true);
        }
        catch (Exception ex)
        {
            logger?.LogWarning(ex, "同步 config_agent.json 失败 path={Path}", agentConfigPath);
            return (false, false);
        }
    }

    public static string ResolveInstallAgentConfigPath(AppSettings settings)
    {
        var rel = settings.TradingAgent?.ConfigPath;
        if (string.IsNullOrWhiteSpace(rel))
            rel = "config/config_agent.json";

        if (Path.IsPathRooted(rel))
            return rel;

        var install = Path.Combine(AppPaths.InstallDir, rel);
        return File.Exists(install) ? install : Path.Combine(AppPaths.InstallDir, "config", "config_agent.json");
    }

    public static string ResolveAgentConfigPath(AppSettings settings)
    {
        var installCfg = ResolveInstallAgentConfigPath(settings);
        var userCfg = UserAgentConfigPath;

        if (!File.Exists(userCfg) && File.Exists(installCfg))
        {
            try
            {
                Directory.CreateDirectory(AppPaths.AppDataDir);
                File.Copy(installCfg, userCfg, overwrite: false);
            }
            catch
            {
                /* 安装目录副本仍可用（只读时无法写入 install） */
            }
        }

        return File.Exists(userCfg) ? userCfg : installCfg;
    }

    public static void EnsureUserAgentConfig(AppSettings settings)
    {
        ForceCopyAgentConfigFromInstall(settings);
    }

    /// <summary>
    /// 安装包 config_agent.json 为权威来源；升级后必须覆盖 AppData，避免 legacy 模板残留。
    /// </summary>
    public static (bool Ok, bool Changed) ForceCopyAgentConfigFromInstall(AppSettings settings, ILogger? logger = null)
    {
        var installCfg = ResolveInstallAgentConfigPath(settings);
        var userCfg = UserAgentConfigPath;
        if (!File.Exists(installCfg))
            return (false, false);

        try
        {
            Directory.CreateDirectory(AppPaths.AppDataDir);
            var changed = !File.Exists(userCfg) || !FilesEqual(installCfg, userCfg);
            File.Copy(installCfg, userCfg, overwrite: true);
            if (changed)
                logger?.LogInformation("已从安装目录覆盖 AppData config_agent.json path={Path}", userCfg);
            return (true, changed);
        }
        catch (Exception ex)
        {
            logger?.LogWarning(ex, "覆盖 AppData config_agent.json 失败");
            return (false, false);
        }
    }

    /// <summary>
    /// 安装版本高于 AppData 时，用安装包 config.json 全覆盖（与安装程序 post_setup 一致）。
    /// </summary>
    public static (bool Ok, bool Changed) ForceCopyMainConfigIfInstallUpgraded(ILogger? logger = null)
    {
        var installMain = Path.Combine(AppPaths.InstallDir, "config.json");
        var userMain = Path.Combine(AppPaths.AppDataDir, "config.json");
        if (!File.Exists(installMain))
            return (false, false);

        var installVer = ReadConfigAppVersion(installMain);
        var userVer = File.Exists(userMain) ? ReadConfigAppVersion(userMain) : "";
        // 仅当 AppData 版本高于安装包时保留用户配置；同版本重装/漂移则覆盖
        if (!string.IsNullOrWhiteSpace(userVer) &&
            string.Compare(installVer, userVer, StringComparison.OrdinalIgnoreCase) < 0)
            return (true, false);

        try
        {
            Directory.CreateDirectory(AppPaths.AppDataDir);
            File.Copy(installMain, userMain, overwrite: true);
            logger?.LogInformation("安装版本 {InstallVer} 已同步 AppData config.json（原 {UserVer}）",
                installVer, string.IsNullOrWhiteSpace(userVer) ? "(无)" : userVer);
            return (true, true);
        }
        catch (Exception ex)
        {
            logger?.LogWarning(ex, "覆盖 AppData config.json 失败");
            return (false, false);
        }
    }

    private static string ReadConfigAppVersion(string path)
    {
        try
        {
            var text = File.ReadAllText(path);
            if (text.Length > 0 && text[0] == '\uFEFF')
                text = text[1..];
            using var doc = JsonDocument.Parse(text);
            if (doc.RootElement.TryGetProperty("app", out var app) &&
                app.TryGetProperty("version", out var ver))
                return ver.GetString() ?? "";
        }
        catch
        {
            /* ignore */
        }
        return "";
    }

    private static bool FilesEqual(string a, string b)
    {
        try
        {
            var fa = new FileInfo(a);
            var fb = new FileInfo(b);
            if (fa.Length != fb.Length)
                return false;
            return File.ReadAllBytes(a).AsSpan().SequenceEqual(File.ReadAllBytes(b));
        }
        catch
        {
            return false;
        }
    }

    public static (bool Ok, bool Changed) AlignWithAppSettings(AppSettings settings, ILogger? logger = null)
    {
        if (settings.TradingAgent is null)
            return (false, false);

        EnsureUserAgentConfig(settings);
        var enabled = settings.TradingAgent.Enabled;
        // 始终写入 AppData；Program Files 下 config 常只读，写 install 路径会静默失败
        var path = File.Exists(UserAgentConfigPath)
            ? UserAgentConfigPath
            : ResolveAgentConfigPath(settings);
        var (knOk, knChanged) = TrySyncKn1FromInstall(settings, logger);
        var (rtOk, rtChanged) = TrySyncRuntimeFieldsFromInstall(settings, logger);
        var (enOk, enChanged) = TrySyncEnabled(path, enabled, logger);
        var (exOk, exChanged) = EnforceExclusiveAgentMode(settings, logger);
        var (expOk, expChanged) = TrySyncSignalExpiry(settings, logger);
        return (enOk && knOk && rtOk && exOk && expOk, enChanged || knChanged || rtChanged || exChanged || expChanged);
    }

    /// <summary>智能体开启时关闭 multi_strategy，避免双模式配置误导。</summary>
    public static (bool Ok, bool Changed) EnforceExclusiveAgentMode(AppSettings settings, ILogger? logger = null)
    {
        if (settings.TradingAgent?.Enabled != true)
            return (true, false);
        if (settings.MultiStrategy is null)
            return (true, false);
        if (!settings.MultiStrategy.Enabled)
            return (true, false);
        settings.MultiStrategy.Enabled = false;
        try
        {
            settings.Save(AppPaths.ConfigPath);
            logger?.LogInformation("智能体模式已开启，已自动关闭 multi_strategy.enabled");
        }
        catch (Exception ex)
        {
            logger?.LogWarning(ex, "写入 config.json multi_strategy 失败");
        }
        return (true, true);
    }

    /// <summary>将 config.json signal_expiry_minutes 同步到 config_agent.json。</summary>
    public static (bool Ok, bool Changed) TrySyncSignalExpiry(AppSettings settings, ILogger? logger = null)
    {
        var userCfg = UserAgentConfigPath;
        if (!File.Exists(userCfg))
            return (false, false);
        var mins = settings.SignalFilters?.SignalExpiryMinutes ?? 240;
        if (mins <= 0)
            return (true, false);
        try
        {
            var text = File.ReadAllText(userCfg);
            if (text.Length > 0 && text[0] == '\uFEFF')
                text = text[1..];
            var node = JsonNode.Parse(text);
            if (node is not JsonObject root)
                return (false, false);
            if (root.TryGetPropertyValue("signal_expiry_minutes", out var cur) && cur is JsonValue jv &&
                jv.TryGetValue<int>(out var curMins) && curMins == mins)
                return (true, false);
            root["signal_expiry_minutes"] = mins;
            File.WriteAllText(userCfg, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }),
                new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            logger?.LogInformation("已同步 config_agent signal_expiry_minutes={Mins}", mins);
            return (true, true);
        }
        catch (Exception ex)
        {
            logger?.LogWarning(ex, "同步 signal_expiry_minutes 失败");
            return (false, false);
        }
    }

    public static int ResolveAgentSignalExpiryMinutes(AppSettings settings, int fallback = 240)
    {
        var path = ResolveAgentConfigPath(settings);
        if (!File.Exists(path))
            return settings.SignalFilters?.SignalExpiryMinutes ?? fallback;
        try
        {
            var text = File.ReadAllText(path);
            if (text.Length > 0 && text[0] == '\uFEFF')
                text = text[1..];
            using var doc = JsonDocument.Parse(text);
            if (doc.RootElement.TryGetProperty("signal_expiry_minutes", out var el) &&
                el.TryGetInt32(out var mins) && mins > 0)
                return mins;
        }
        catch
        {
            /* fallback */
        }
        return settings.SignalFilters?.SignalExpiryMinutes ?? fallback;
    }

    /// <summary>
    /// 将安装包 V16 运行时关键字段合并到 AppData config_agent.json。
    /// </summary>
    public static (bool Ok, bool Changed) TrySyncRuntimeFieldsFromInstall(AppSettings settings, ILogger? logger = null)
    {
        var installCfg = ResolveInstallAgentConfigPath(settings);
        var userCfg = UserAgentConfigPath;
        if (!File.Exists(installCfg) || !File.Exists(userCfg))
            return (false, false);

        try
        {
            var installRoot = JsonNode.Parse(File.ReadAllText(installCfg)) as JsonObject;
            var userRoot = JsonNode.Parse(File.ReadAllText(userCfg)) as JsonObject;
            if (installRoot is null || userRoot is null)
                return (false, false);

            var changed = false;
            foreach (var key in new[]
                     {
                         "architecture", "causal", "counterfactual", "execution_gates", "execution_composer",
                         "execution_composer_v17", "trader_mind", "rl_inference", "trading_env",
                     })
            {
                if (!installRoot.TryGetPropertyValue(key, out var installVal) || installVal is null)
                    continue;
                if (!JsonNodesEqual(userRoot[key], installVal))
                {
                    userRoot[key] = installVal.DeepClone();
                    changed = true;
                }
            }

            changed |= SyncArchitectureV17FromInstall(installRoot, userRoot);
            changed |= SyncKn2ContractFromInstall(installRoot, userRoot);

            if (!changed)
                return (true, false);

            File.WriteAllText(userCfg, userRoot.ToJsonString(new JsonSerializerOptions { WriteIndented = true }),
                new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            logger?.LogInformation("已同步 config_agent.json 运行时字段 path={Path}", userCfg);
            return (true, true);
        }
        catch (Exception ex)
        {
            logger?.LogWarning(ex, "同步 config_agent.json 运行时字段失败");
            return (false, false);
        }
    }

    /// <summary>
    /// 将安装包中 KN1 关键字段合并到 AppData config_agent.json，避免旧版 KN2 配置残留。
    /// 仅覆盖 kn2.enabled 与 knowledge_net.input_dim（安装包为准）。
    /// </summary>
    public static (bool Ok, bool Changed) TrySyncKn1FromInstall(AppSettings settings, ILogger? logger = null)
    {
        var installCfg = ResolveInstallAgentConfigPath(settings);
        var userCfg = UserAgentConfigPath;
        if (!File.Exists(installCfg) || !File.Exists(userCfg))
            return (false, false);

        try
        {
            var installText = File.ReadAllText(installCfg);
            if (installText.Length > 0 && installText[0] == '\uFEFF')
                installText = installText[1..];
            var userText = File.ReadAllText(userCfg);
            if (userText.Length > 0 && userText[0] == '\uFEFF')
                userText = userText[1..];

            var installRoot = JsonNode.Parse(installText) as JsonObject;
            var userRoot = JsonNode.Parse(userText) as JsonObject;
            if (installRoot is null || userRoot is null)
                return (false, false);

            var changed = false;

            if (installRoot.TryGetPropertyValue("kn2", out var installKn2) && installKn2 is JsonObject kn2Obj &&
                kn2Obj.TryGetPropertyValue("enabled", out var kn2Enabled))
            {
                var userKn2 = userRoot["kn2"] as JsonObject ?? new JsonObject();
                if (userRoot["kn2"] is null)
                    userRoot["kn2"] = userKn2;

                // 只从安装包「启用」KN2，禁止用旧包 false 覆盖用户 LIVE
                if (kn2Enabled is JsonValue kv && kv.TryGetValue<bool>(out var installOn) && installOn)
                {
                    if (!JsonNodesEqual(userKn2["enabled"], kn2Enabled))
                    {
                        userKn2["enabled"] = kn2Enabled.DeepClone();
                        changed = true;
                    }
                }
            }

            if (installRoot.TryGetPropertyValue("knowledge_net", out var installKn) && installKn is JsonObject knNet &&
                knNet.TryGetPropertyValue("input_dim", out var inputDim))
            {
                var userKn = userRoot["knowledge_net"] as JsonObject ?? new JsonObject();
                if (userRoot["knowledge_net"] is null)
                    userRoot["knowledge_net"] = userKn;
                if (!JsonNodesEqual(userKn["input_dim"], inputDim))
                {
                    userKn["input_dim"] = inputDim?.DeepClone();
                    changed = true;
                }
            }

            if (!changed)
                return (true, false);

            var outText = userRoot.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(userCfg, outText, new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            logger?.LogInformation("已同步 config_agent.json KN1 字段 path={Path}", userCfg);
            return (true, true);
        }
        catch (Exception ex)
        {
            logger?.LogWarning(ex, "同步 config_agent.json KN1 字段失败");
            return (false, false);
        }
    }

    /// <summary>
    /// 安装包含 V17 子段时，补全 AppData architecture.direction_scorer / location_gate（避免仅升部分字段）。
    /// </summary>
    private static bool SyncArchitectureV17FromInstall(JsonObject installRoot, JsonObject userRoot)
    {
        if (!installRoot.TryGetPropertyValue("architecture", out var installArchNode) ||
            installArchNode is not JsonObject installArch)
            return false;

        var userArch = userRoot["architecture"] as JsonObject ?? new JsonObject();
        if (userRoot["architecture"] is null)
            userRoot["architecture"] = userArch;

        var changed = false;
        foreach (var subKey in new[] { "version", "direction_scorer", "location_gate", "horizon_predictor" })
        {
            if (!installArch.TryGetPropertyValue(subKey, out var installSub) || installSub is null)
                continue;
            if (!JsonNodesEqual(userArch[subKey], installSub))
            {
                userArch[subKey] = installSub.DeepClone();
                changed = true;
            }
        }

        return changed;
    }

    private static bool SyncKn2ContractFromInstall(JsonObject installRoot, JsonObject userRoot)
    {
        if (!installRoot.TryGetPropertyValue("kn2", out var installKn2) || installKn2 is not JsonObject installKn2Obj)
            return false;

        var userKn2 = userRoot["kn2"] as JsonObject ?? new JsonObject();
        if (userRoot["kn2"] is null)
            userRoot["kn2"] = userKn2;

        var wasLive = userKn2.TryGetPropertyValue("enabled", out var userEnabled) &&
                      userEnabled is JsonValue uev && uev.TryGetValue<bool>(out var liveOn) && liveOn &&
                      !(userKn2.TryGetPropertyValue("shadow_mode", out var userShadow) &&
                        userShadow is JsonValue usv && usv.TryGetValue<bool>(out var shadowOn) && shadowOn);

        var changed = false;
        foreach (var prop in new[] { "model_path", "min_confidence", "enabled", "shadow_mode" })
        {
            if (!installKn2Obj.TryGetPropertyValue(prop, out var installVal))
                continue;
            if (wasLive && prop is "enabled" or "shadow_mode")
                continue;
            if (!JsonNodesEqual(userKn2[prop], installVal))
            {
                userKn2[prop] = installVal?.DeepClone();
                changed = true;
            }
        }

        if (wasLive)
        {
            if (userKn2["enabled"] is not JsonValue en || !en.TryGetValue<bool>(out var on) || !on)
            {
                userKn2["enabled"] = true;
                changed = true;
            }
            if (userKn2["shadow_mode"] is not JsonValue sm || !sm.TryGetValue<bool>(out var sh) || sh)
            {
                userKn2["shadow_mode"] = false;
                changed = true;
            }
        }

        return changed;
    }

    private static bool JsonNodesEqual(JsonNode? a, JsonNode? b)
    {
        if (a is null && b is null) return true;
        if (a is null || b is null) return false;
        return a.ToJsonString() == b.ToJsonString();
    }
}
