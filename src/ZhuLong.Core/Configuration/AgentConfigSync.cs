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
        var userCfg = UserAgentConfigPath;
        if (File.Exists(userCfg) && new FileInfo(userCfg).Length > 32)
            return;

        var installCfg = ResolveInstallAgentConfigPath(settings);
        if (!File.Exists(installCfg))
            return;

        try
        {
            Directory.CreateDirectory(AppPaths.AppDataDir);
            File.Copy(installCfg, userCfg, overwrite: true);
        }
        catch
        {
            /* 安装目录副本仍可读 */
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
        var (enOk, enChanged) = TrySyncEnabled(path, enabled, logger);
        return (enOk && knOk, enChanged || knChanged);
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
                if (!JsonNodesEqual(userKn2["enabled"], kn2Enabled))
                {
                    userKn2["enabled"] = kn2Enabled?.DeepClone();
                    changed = true;
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

    private static bool JsonNodesEqual(JsonNode? a, JsonNode? b)
    {
        if (a is null && b is null) return true;
        if (a is null || b is null) return false;
        return a.ToJsonString() == b.ToJsonString();
    }
}
