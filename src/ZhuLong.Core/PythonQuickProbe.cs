using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace ZhuLong.Core;

/// <summary>解析 Python 并做子进程自检（不依赖 GUI 进程 PATH）。</summary>
public static class PythonQuickProbe
{
    public static bool TryRunVersionLine(string configuredPython, out string? resolvedExe, out string? versionLine, out string? error)
    {
        resolvedExe = null;
        versionLine = null;
        error = null;

        if (!PythonExecutableResolver.TryResolve(configuredPython, out resolvedExe, out error))
            return false;

        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = resolvedExe,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };

            if (IsPyLauncher(resolvedExe))
                psi.ArgumentList.Add("-3");
            psi.ArgumentList.Add("-c");
            psi.ArgumentList.Add("import sys; print(sys.version)");

            using var p = Process.Start(psi);
            if (p is null)
            {
                error = "无法启动 Python 子进程。";
                return false;
            }

            versionLine = p.StandardOutput.ReadToEnd().Trim();
            var err = p.StandardError.ReadToEnd().Trim();
            p.WaitForExit(20000);
            if (p.ExitCode != 0)
            {
                error = string.IsNullOrEmpty(err) ? "退出码 " + p.ExitCode : err;
                return false;
            }

            return !string.IsNullOrEmpty(versionLine);
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return false;
        }
    }

    public static bool TryImportModule(string configuredPython, string moduleName, out string? error)
    {
        error = null;
        if (!PythonExecutableResolver.TryResolve(configuredPython, out var exe, out error))
            return false;

        var lit = PythonModuleLiteral(moduleName);
        var code = $"import importlib; importlib.import_module({lit}); print('OK')";
        return RunPythonCode(exe, code, moduleName + " import 失败", out error);
    }

    /// <summary>读取 numpy 版本（RL 权重需 numpy 2.x）。</summary>
    public static bool TryGetNumpyVersion(string configuredPython, out Version? version, out string? error)
    {
        version = null;
        error = null;
        if (!PythonExecutableResolver.TryResolve(configuredPython, out var exe, out error))
            return false;

        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = exe,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            if (IsPyLauncher(exe))
                psi.ArgumentList.Add("-3");
            psi.ArgumentList.Add("-c");
            psi.ArgumentList.Add("import numpy; print(numpy.__version__)");

            using var p = Process.Start(psi);
            if (p is null)
            {
                error = "无法启动 Python。";
                return false;
            }

            var verText = p.StandardOutput.ReadToEnd().Trim();
            var err = p.StandardError.ReadToEnd().Trim();
            p.WaitForExit(20000);
            if (p.ExitCode != 0 || string.IsNullOrEmpty(verText))
            {
                error = string.IsNullOrEmpty(err) ? "numpy 版本读取失败" : err;
                return false;
            }

            version = Version.Parse(verText.Split('+')[0].Trim());
            return true;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return false;
        }
    }

    public static bool TryImportSmoke(string configuredPython, out string? error)
    {
        error = null;
        if (!PythonExecutableResolver.TryResolve(configuredPython, out var exe, out error))
            return false;

        const string code = "import torch,xgboost,pandas,pyarrow,MetaTrader5,sklearn,joblib; print('OK')";
        return RunPythonCode(exe, code, "依赖 import 失败", out error);
    }

    /// <summary>
    /// 与 Worker 相同路径执行 inference_cli agent_validate（Horizon Session + scaler + 热更新探针）。
    /// 设置页「import onnxruntime」通过但本探针失败时，多为 VC++ 运行库或 onnxruntime 原生 DLL 问题。
    /// </summary>
    public static bool TryRunAgentValidateCli(
        string configuredPython,
        string installDir,
        string configPath,
        out string? error)
    {
        error = null;
        var cli = AppPaths.InferenceCliScriptPath;
        if (!File.Exists(cli))
        {
            error = "missing_inference_cli:" + cli;
            return false;
        }

        if (!PythonExecutableResolver.TryResolve(configuredPython, out var exe, out error))
            return false;

        var input = Path.Combine(Path.GetTempPath(), "zhulong_val_" + Guid.NewGuid().ToString("N") + ".json");
        var output = Path.Combine(Path.GetTempPath(), "zhulong_val_out_" + Guid.NewGuid().ToString("N") + ".json");
        try
        {
            var req = new Dictionary<string, object?>
            {
                ["cmd"] = "agent_validate",
                ["root"] = installDir,
                ["config_path"] = configPath,
                ["quick"] = true,
            };
            File.WriteAllText(input, JsonSerializer.Serialize(req), new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

            var psi = new ProcessStartInfo
            {
                FileName = exe,
                WorkingDirectory = installDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            if (IsPyLauncher(exe))
                psi.ArgumentList.Add("-3");
            psi.ArgumentList.Add(cli);
            psi.ArgumentList.Add("--input");
            psi.ArgumentList.Add(input);
            psi.ArgumentList.Add("--output");
            psi.ArgumentList.Add(output);
            ApplyCliSubprocessEnv(psi, installDir);

            using var p = Process.Start(psi);
            if (p is null)
            {
                error = "无法启动 inference_cli 子进程。";
                return false;
            }

            var stderr = p.StandardError.ReadToEnd().Trim();
            _ = p.StandardOutput.ReadToEnd();
            p.WaitForExit(120000);

            if (!File.Exists(output))
            {
                error = string.IsNullOrEmpty(stderr)
                    ? "agent_validate 无输出（退出码 " + p.ExitCode + "）"
                    : stderr;
                return false;
            }

            using var doc = JsonDocument.Parse(File.ReadAllText(output));
            var root = doc.RootElement;
            if (root.TryGetProperty("ok", out var okEl) && okEl.ValueKind == JsonValueKind.True)
                return true;

            if (root.TryGetProperty("error", out var errEl))
                error = errEl.GetString();
            else
                error = string.IsNullOrEmpty(stderr) ? "agent_validate 失败" : stderr;
            return false;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return false;
        }
        finally
        {
            try
            {
                if (File.Exists(input)) File.Delete(input);
                if (File.Exists(output)) File.Delete(output);
            }
            catch
            {
                // ignore temp cleanup
            }
        }
    }

    /// <summary>尝试创建 ONNX InferenceSession（仅 raw ONNX，不含 Worker 热更新路径；优先用 TryRunAgentValidateCli）。</summary>
    public static bool TryLoadHorizonOnnx(string configuredPython, string onnxFullPath, string installDir, out string? error)
    {
        error = null;
        if (!File.Exists(onnxFullPath))
        {
            error = "onnx_not_found:" + onnxFullPath;
            return false;
        }

        if (new FileInfo(onnxFullPath).Length < 4096)
        {
            error = "horizon_onnx_invalid:size=" + new FileInfo(onnxFullPath).Length;
            return false;
        }

        if (!PythonExecutableResolver.TryResolve(configuredPython, out var exe, out error))
            return false;

        var onnxLit = PythonStringLiteral(onnxFullPath);
        var installLit = PythonStringLiteral(installDir);
        var appDataLit = PythonStringLiteral(AppPaths.AppDataDir);
        var engineLit = PythonStringLiteral(AppPaths.PythonEngineDir);
        var code =
            "import os,sys; from pathlib import Path; " +
            "install=Path(" + installLit + "); appdata=Path(" + appDataLit + "); engine=Path(" + engineLit + "); " +
            "os.environ['ZHULONG_INSTALL_DIR']=str(install); " +
            "for p in reversed([install, engine, appdata if appdata.is_dir() else None]): " +
            "  s=str(p) if p and p.is_dir() else None; " +
            "  if s and s not in sys.path: sys.path.insert(0,s); " +
            "from zhulong.utils.win_dll import configure_native_dll_paths; configure_native_dll_paths(); " +
            "if engine.is_dir() and str(engine) not in sys.path: sys.path.insert(0,str(engine)); " +
            "from hotfix_loader import apply_appdata_hotfixes; apply_appdata_hotfixes(); " +
            "import onnxruntime as ort; " +
            "ort.InferenceSession(" + onnxLit + ", ort.SessionOptions(), providers=['CPUExecutionProvider']); " +
            "print('OK')";
        return RunPythonCode(exe, code, "Horizon ONNX Session 创建失败", out error);
    }

    private static void ApplyCliSubprocessEnv(ProcessStartInfo psi, string installDir)
    {
        var appData = AppPaths.AppDataDir;
        var engine = AppPaths.PythonEngineDir;
        var bundled = AppPaths.BundledPythonDir;
        psi.Environment["ZHULONG_INSTALL_DIR"] = installDir;
        psi.Environment["PYTHONDONTWRITEBYTECODE"] = "1";
        psi.Environment["PYTHONIOENCODING"] = "utf-8";
        psi.Environment["TF_CPP_MIN_LOG_LEVEL"] = "3";
        if (AppPaths.HasBundledPython)
        {
            psi.Environment["ZHULONG_BUNDLED_PYTHON"] = "1";
            if (PythonRuntime.TryResolveBundled(out _, out var dll) && !string.IsNullOrEmpty(dll))
                psi.Environment["PYTHONNET_PYDLL"] = dll;
        }
        var paths = new List<string> { appData, installDir, engine };
        if (Directory.Exists(bundled))
            paths.Insert(0, bundled);
        psi.Environment["PYTHONPATH"] = string.Join(Path.PathSeparator.ToString(), paths);
        var pathPrefix = new List<string>();
        if (Directory.Exists(bundled))
            pathPrefix.Add(bundled);
        pathPrefix.Add(installDir);
        if (psi.Environment.TryGetValue("PATH", out var cur) && !string.IsNullOrWhiteSpace(cur))
            pathPrefix.Add(cur);
        psi.Environment["PATH"] = string.Join(Path.PathSeparator.ToString(), pathPrefix);
    }

    private static string PythonStringLiteral(string value) =>
        "r'" + value.Replace("\\", "\\\\", StringComparison.Ordinal)
            .Replace("'", "\\'", StringComparison.Ordinal) + "'";

    private static string PythonModuleLiteral(string moduleName)
    {
        return "'" + moduleName.Replace("\\", "\\\\", StringComparison.Ordinal)
            .Replace("'", "\\'", StringComparison.Ordinal) + "'";
    }

    private static bool RunPythonCode(string exe, string code, string fallbackError, out string? error)
    {
        error = null;
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = exe,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            if (IsPyLauncher(exe))
                psi.ArgumentList.Add("-3");
            psi.ArgumentList.Add("-c");
            psi.ArgumentList.Add(code);

            using var p = Process.Start(psi);
            if (p is null)
            {
                error = "无法启动 Python。";
                return false;
            }

            var err = p.StandardError.ReadToEnd().Trim();
            _ = p.StandardOutput.ReadToEnd();
            p.WaitForExit(120000);
            if (p.ExitCode != 0)
            {
                error = string.IsNullOrEmpty(err) ? fallbackError + "（退出码 " + p.ExitCode + "）" : err;
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return false;
        }
    }

    public static bool IsPyLauncher(string exe) =>
        Path.GetFileName(exe).Equals("py.exe", StringComparison.OrdinalIgnoreCase);
}
