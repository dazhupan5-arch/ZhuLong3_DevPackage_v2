using System.Diagnostics;
using System.Text;
using ZhuLong.Core;
using ZhuLong.Core.Configuration;
using ZhuLong.Core.Services;

namespace ZhuLong.App.Services;

/// <summary>触发 Python 离线宏观拉取并刷新 C# 缓存。</summary>
public sealed class MacroOfflineRefreshService
{
    private readonly MacroCalendarService _macro;
    private readonly UserSecretsStore _secrets;

    public MacroOfflineRefreshService(MacroCalendarService macro, UserSecretsStore secrets)
    {
        _macro = macro;
        _secrets = secrets;
    }

    public async Task<(bool Ok, string Message)> RefreshAllAsync(CancellationToken ct = default)
    {
        PythonRuntime.InvalidateCache();
        PythonRuntime.DiscoverAndCache(force: true);

        var root = AppPaths.FindDevRoot() ?? AppPaths.InstallDir;
        var py = PythonRuntime.ResolveExecutable();
        if (!File.Exists(py) && py is not "py" and not "python")
        {
            return (false, "未找到 Python。请在设置页点「一键修复环境」。");
        }

        var errors = new List<string>();
        foreach (var script in new[] { "fetch_fred.py", "fetch_sentiment.py" })
        {
            var path = Path.Combine(root, "ZhuLong.PythonEngine", script);
            if (!File.Exists(path))
            {
                errors.Add($"{script}: 文件不存在");
                continue;
            }

            var (code, err) = await RunPythonAsync(py, path, root, ct).ConfigureAwait(false);
            if (code != 0)
                errors.Add($"{script}: {TrimError(err)}");
        }

        _macro.ReloadOfflineJson();
        await _macro.RefreshCalendarAsync(ct).ConfigureAwait(false);

        if (errors.Count > 0)
            return (false, string.Join(Environment.NewLine, errors));
        return (true, "宏观离线数据与日历已刷新");
    }

    private static string TrimError(string err)
    {
        if (string.IsNullOrWhiteSpace(err)) return "未知错误";
        var lines = err.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        foreach (var line in lines)
        {
            if (line.Contains("PermissionError", StringComparison.Ordinal) ||
                line.Contains("ModuleNotFoundError", StringComparison.Ordinal) ||
                line.Contains("ImportError", StringComparison.Ordinal) ||
                line.Contains("请安装", StringComparison.Ordinal))
                return line;
        }
        return lines.Length > 0 ? lines[^1] : err.Trim();
    }

    private async Task<(int Code, string Err)> RunPythonAsync(
        string py, string script, string cwd, CancellationToken ct)
    {
        if (!PythonExecutableResolver.TryResolve(py, out var exe, out var resolveErr))
            return (1, resolveErr ?? "Python 解析失败");

        var psi = new ProcessStartInfo
        {
            FileName = exe,
            WorkingDirectory = cwd,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };

        if (PythonQuickProbe.IsPyLauncher(exe))
            psi.ArgumentList.Add("-3");
        psi.ArgumentList.Add(script);

        ApplyPythonEnv(psi);

        using var p = Process.Start(psi)!;
        var stdout = await p.StandardOutput.ReadToEndAsync(ct).ConfigureAwait(false);
        var err = await p.StandardError.ReadToEndAsync(ct).ConfigureAwait(false);
        await p.WaitForExitAsync(ct).ConfigureAwait(false);
        var combined = string.IsNullOrWhiteSpace(err) ? stdout : err + Environment.NewLine + stdout;
        return (p.ExitCode, combined);
    }

    private void ApplyPythonEnv(ProcessStartInfo psi)
    {
        var dll = AppPaths.FindPythonDll();
        if (!string.IsNullOrEmpty(dll))
            psi.Environment["PYTHONNET_PYDLL"] = dll;

        psi.Environment["ZHULONG_DATA_DIR"] = AppPaths.WritableDataDir;
        psi.Environment["ZHULONG_LOGS_DIR"] = AppPaths.LogsDir;
        psi.Environment["APPDATA"] = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);

        if (_secrets.ResolveFredApiKey() is { } fred)
            psi.Environment["FRED_API_KEY"] = fred;
        if (_secrets.ResolveFinnhubApiKey() is { } fh)
            psi.Environment["FINNHUB_API_KEY"] = fh;
        if (_secrets.ResolveFmpApiKey() is { } fmp)
            psi.Environment["FMP_API_KEY"] = fmp;
        if (_secrets.ResolveLlmApiKey() is { } llm)
        {
            psi.Environment["LLM_API_KEY"] = llm;
            psi.Environment["GEMINI_API_KEY"] = llm;
            psi.Environment["API2D_API_KEY"] = llm;
        }
    }
}
