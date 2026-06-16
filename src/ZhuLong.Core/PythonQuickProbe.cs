using System.Diagnostics;

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

    public static bool TryImportSmoke(string configuredPython, out string? error)
    {
        error = null;
        if (!PythonExecutableResolver.TryResolve(configuredPython, out var exe, out error))
            return false;

        const string code = "import torch,xgboost,pandas,pyarrow,MetaTrader5,sklearn,joblib; print('OK')";
        return RunPythonCode(exe, code, "依赖 import 失败", out error);
    }

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
