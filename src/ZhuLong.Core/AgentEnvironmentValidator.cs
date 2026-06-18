using System.Text.Json;
using System.Text.Json.Nodes;

namespace ZhuLong.Core;

/// <summary>V16 智能体实机环境：安装目录文件 + Python 模块 + inference_cli agent_validate（与 Worker 同路径探针）。</summary>
public static class AgentEnvironmentValidator
{
    public static bool TryValidateV16(string configPath, string? pythonExe, out string? error)
    {
        error = null;
        if (!File.Exists(configPath))
        {
            error = "config_not_found:" + configPath;
            return false;
        }

        JsonObject? root;
        try
        {
            var text = File.ReadAllText(configPath);
            if (text.Length > 0 && text[0] == '\uFEFF')
                text = text[1..];
            root = JsonNode.Parse(text) as JsonObject;
        }
        catch (Exception ex)
        {
            error = "config_parse:" + ex.Message;
            return false;
        }

        if (root is null)
        {
            error = "config_invalid";
            return false;
        }

        if (root.TryGetPropertyValue("enabled", out var en) && en is JsonValue ev &&
            ev.TryGetValue<bool>(out var enabled) && !enabled)
        {
            error = "agent_disabled_in_config";
            return false;
        }

        var arch = (root["architecture"] as JsonObject)?["version"]?.GetValue<string>() ?? "legacy";
        if (!string.Equals(arch, "v16", StringComparison.OrdinalIgnoreCase))
        {
            error = "architecture_not_v16:" + arch;
            return false;
        }

        var hp = (root["architecture"] as JsonObject)?["horizon_predictor"] as JsonObject;
        var horizonOnnx = hp?["model_path"]?.GetValue<string>() ?? "models/horizon_v16.onnx";
        var horizonScaler = hp?["scaler_path"]?.GetValue<string>() ?? "models/horizon_v16_scaler.pkl";
        var onnxFile = ResolveBundledFile(horizonOnnx);
        foreach (var rel in new[] { horizonOnnx, horizonScaler })
        {
            if (!ResolveBundledFile(rel).Exists)
            {
                error = "missing_horizon:" + rel;
                return false;
            }
        }

        if (onnxFile.Length < 4096)
        {
            error = "horizon_onnx_invalid:size=" + onnxFile.Length + ":" + onnxFile.FullName;
            return false;
        }

        var sym = root["primary_symbol"]?.GetValue<string>() ?? "XAUUSD";
        var useRl = root["use_rl"]?.GetValue<bool>() == true;
        if (useRl && !ResolveRlModelExists(root, sym))
        {
            error = "missing_rl:" + ResolveRlModelRel(root, sym);
            return false;
        }

        var kn2 = root["kn2"] as JsonObject;
        if (kn2 is not null &&
            (kn2["enabled"]?.GetValue<bool>() == true || kn2["shadow_mode"]?.GetValue<bool>() == true))
        {
            var kn2Rel = kn2["model_path"]?.GetValue<string>() ?? "models/kn2_trader_v16.pth";
            if (!ResolveBundledFile(kn2Rel).Exists)
            {
                error = "missing_kn2:" + kn2Rel;
                return false;
            }
        }

        var cli = AppPaths.InferenceCliScriptPath;
        if (!File.Exists(cli))
        {
            error = "missing_inference_cli:" + cli;
            return false;
        }

        var py = pythonExe;
        if (string.IsNullOrWhiteSpace(py) || !File.Exists(py))
        {
            if (!PythonExecutableResolver.TryResolve(PythonRuntime.ResolveExecutable(), out py, out var resolveErr))
            {
                error = resolveErr ?? "python_not_found";
                return false;
            }
        }

        if (!PythonQuickProbe.TryRunVersionLine(py!, out _, out var ver, out var verErr))
        {
            error = verErr ?? "python_version_check_failed";
            return false;
        }

        if (ver is not null && ver.Contains("3.9", StringComparison.Ordinal))
        {
            error = "python_too_old:" + ver;
            return false;
        }

        foreach (var mod in new[] { "numpy", "pandas", "sklearn", "joblib", "onnxruntime" })
        {
            if (!PythonQuickProbe.TryImportModule(py!, mod, out var modErr))
            {
                error = "missing_python_module:" + mod + ":" + modErr;
                return false;
            }
        }

        if (!PythonQuickProbe.TryRunAgentValidateCli(py!, AppPaths.InstallDir, configPath, out var probeErr))
        {
            error = probeErr ?? "agent_validate_probe_failed";
            return false;
        }

        if (useRl)
        {
            if (!PythonQuickProbe.TryGetNumpyVersion(py!, out var npVer, out var npErr))
            {
                error = "numpy_version_check_failed:" + npErr;
                return false;
            }

            if (npVer is null || npVer.Major < 2)
            {
                error = "numpy_too_old:" + (npVer?.ToString() ?? "?") + ":rl_requires_numpy>=2";
                return false;
            }

            if (!PythonQuickProbe.TryImportModule(py!, "stable_baselines3", out var sb3Err))
            {
                error = "missing_python_module:stable_baselines3:" + sb3Err;
                return false;
            }
        }

        return true;
    }

    private static string ResolveRlModelRel(JsonObject root, string symbol)
    {
        var sym = symbol.ToUpperInvariant();
        var symBlock = (root["symbols"] as JsonObject)?[sym] as JsonObject;
        var rlSym = symBlock?["rl"] as JsonObject;
        var rlRoot = root["rl"] as JsonObject;
        string? rel = rlSym?["model_path"]?.GetValue<string>();
        if (string.IsNullOrWhiteSpace(rel))
        {
            rel = sym == "USOIL"
                ? rlRoot?["model_path_oil"]?.GetValue<string>() ?? "models/rl_agent_oil"
                : rlRoot?["model_path_xau"]?.GetValue<string>()
                  ?? rlRoot?["model_path"]?.GetValue<string>()
                  ?? "models/rl_agent_xau";
        }
        return rel;
    }

    private static bool ResolveRlModelExists(JsonObject root, string symbol)
    {
        var rel = ResolveRlModelRel(root, symbol);
        var basePath = ResolveBundledFile(rel);
        if (basePath.Exists)
            return true;
        var zipPath = ResolveBundledFile(rel.EndsWith(".zip", StringComparison.OrdinalIgnoreCase) ? rel : rel + ".zip");
        return zipPath.Exists;
    }

    private static FileInfo ResolveBundledFile(string rel)
    {
        var p = Path.IsPathRooted(rel) ? new FileInfo(rel) : new FileInfo(Path.Combine(AppPaths.InstallDir, rel));
        if (p.Exists)
            return p;
        var appData = new FileInfo(Path.Combine(AppPaths.AppDataDir, rel));
        return appData.Exists ? appData : p;
    }
}
