using System.Diagnostics;

namespace ZhuLong.Core.Bootstrap;

/// <summary>
/// MT5 桥接文件部署（已禁用自动部署，避免启动时覆盖/删除用户 MT5 中的指标与 DLL）。
/// 请手动将 <c>mql5\Libraries\ZhuLongMt5Pipe.dll</c> 与指标复制到 MT5 目录，或运行 <c>scripts\deploy-mt5-indicator.ps1</c>。
/// </summary>
public static class Mt5BridgeDeployer
{
    private const string PipeDllName = "ZhuLongMt5Pipe.dll";
    private const string IndicatorMq5 = "ZhuLongIndicator.mq5";
    private const string IndicatorEx5 = "ZhuLongIndicator.ex5";
    private const long MinDllBytes = 10_000;

    /// <summary>自动部署已永久关闭；调用方不应在启动流程中覆盖 MT5 文件。</summary>
    public static bool AutoDeployEnabled => false;

    public sealed record DeployResult(
        IReadOnlyList<string> SuccessRoots,
        IReadOnlyList<string> FailedRoots,
        string? SourceDll,
        string Summary,
        bool NeedsElevation);

    /// <summary>不再自动复制文件到 MT5。仅返回跳过状态，供日志使用。</summary>
    public static DeployResult EnsureDeployed(string? installDir = null, bool allowElevation = true)
    {
        const string summary = "MT5 自动部署已禁用（请手动维护 ZhuLongMt5Pipe.dll 与 ZhuLongIndicator）";
        WriteLog(summary, "INFO");
        return new DeployResult(
            Array.Empty<string>(),
            Array.Empty<string>(),
            null,
            summary,
            false);
    }

    /// <summary>仅供手动脚本调用；烛龙启动流程不会执行。</summary>
    public static DeployResult DeployManually(string? installDir = null, bool allowElevation = true)
    {
        installDir ??= AppPaths.InstallDir;

        var result = DeployOnce(installDir);
        if (result.FailedRoots.Count == 0 || !allowElevation)
            return result;

        WriteLog("Some targets failed; requesting elevated deploy...", "WARN");
        if (TryElevatedDeploy(installDir))
        {
            var retry = DeployOnce(installDir);
            if (retry.FailedRoots.Count == 0)
                return retry with { Summary = "MT5 bridge deployed (elevated)." };
            return retry with { NeedsElevation = true };
        }

        return result with
        {
            NeedsElevation = true,
            Summary = result.Summary + " Run DeployMt5Bridge.cmd as administrator."
        };
    }

    private static DeployResult DeployOnce(string installDir)
    {
        var sources = ResolveBundledSources(installDir);
        var roots = DiscoverMt5Roots();
        var ok = new List<string>();
        var fail = new List<string>();

        if (sources.DllPath is null || !File.Exists(sources.DllPath))
        {
            var msg = $"Install dir missing {PipeDllName}: {installDir}\\mql5\\Libraries\\";
            WriteLog(msg, "ERR");
            return new DeployResult(ok, fail, null, msg, false);
        }

        if (sources.Mq5Path is null || !File.Exists(sources.Mq5Path))
        {
            var msg = $"Install dir missing {IndicatorMq5}";
            WriteLog(msg, "ERR");
            return new DeployResult(ok, fail, sources.DllPath, msg, false);
        }

        if (roots.Count == 0)
        {
            var msg = "No MT5 folder found. Install MT5 first.";
            WriteLog(msg, "WARN");
            return new DeployResult(ok, fail, sources.DllPath, msg, false);
        }

        foreach (var root in roots)
        {
            try
            {
                DeployToRoot(root, sources);
                if (!VerifyPipeDll(root))
                    throw new IOException($"{PipeDllName} not present after copy (access denied or antivirus removed file)");
                ok.Add(root);
                WriteLog($"OK -> {root}");
            }
            catch (Exception ex)
            {
                fail.Add(root);
                WriteLog($"FAIL -> {root}: {ex.Message}", "ERR");
            }
        }

        var summary = fail.Count == 0
            ? $"MT5 bridge OK ({ok.Count} terminal(s))."
            : $"MT5 bridge: {ok.Count} ok, {fail.Count} failed.";
        WriteLog(summary);
        return new DeployResult(ok, fail, sources.DllPath, summary, fail.Count > 0);
    }

    public static bool VerifyPipeDll(string mt5Root)
    {
        var path = Path.Combine(mt5Root, "MQL5", "Libraries", PipeDllName);
        return File.Exists(path) && new FileInfo(path).Length >= MinDllBytes;
    }

    public static bool TryElevatedDeploy(string installDir)
    {
        var script = Path.Combine(installDir, "scripts", "deploy-mt5-indicator.ps1");
        var cmd = Path.Combine(installDir, "DeployMt5Bridge.cmd");
        if (!File.Exists(script))
        {
            WriteLog($"Missing deploy script: {script}", "ERR");
            return false;
        }

        try
        {
            if (File.Exists(cmd))
            {
                var p = Process.Start(new ProcessStartInfo
                {
                    FileName = cmd,
                    WorkingDirectory = installDir,
                    UseShellExecute = true,
                    Verb = "runas",
                });
                p?.WaitForExit(120_000);
                return p is { ExitCode: 0 };
            }

            var ps = Process.Start(new ProcessStartInfo
            {
                FileName = "powershell.exe",
                Arguments = $"-NoProfile -ExecutionPolicy Bypass -File \"{script}\" -InstallDir \"{installDir}\"",
                WorkingDirectory = installDir,
                UseShellExecute = true,
                Verb = "runas",
            });
            ps?.WaitForExit(120_000);
            return ps is { ExitCode: 0 };
        }
        catch (Exception ex)
        {
            WriteLog($"Elevated deploy cancelled or failed: {ex.Message}", "WARN");
            return false;
        }
    }

    private sealed record BundledSources(string? DllPath, string? Mq5Path, string? Ex5Path);

    private static BundledSources ResolveBundledSources(string installDir)
    {
        var dll = Path.Combine(installDir, "mql5", "Libraries", PipeDllName);
        var mq5 = Path.Combine(installDir, "mql5", IndicatorMq5);
        var ex5 = Path.Combine(installDir, "mql5", IndicatorEx5);

        if (!File.Exists(mq5))
            mq5 = Path.Combine(installDir, "indicators", IndicatorMq5);
        if (!File.Exists(ex5))
            ex5 = Path.Combine(installDir, "indicators", IndicatorEx5);

        return new BundledSources(
            File.Exists(dll) ? dll : null,
            File.Exists(mq5) ? mq5 : null,
            File.Exists(ex5) ? ex5 : null);
    }

    private static void DeployToRoot(string mt5Root, BundledSources sources)
    {
        var lib = Path.Combine(mt5Root, "MQL5", "Libraries");
        var ind = Path.Combine(mt5Root, "MQL5", "Indicators");
        Directory.CreateDirectory(lib);
        Directory.CreateDirectory(ind);

        CopyVerified(sources.DllPath!, Path.Combine(lib, PipeDllName));

        CopyVerified(sources.Mq5Path!, Path.Combine(ind, IndicatorMq5));
        if (sources.Ex5Path is not null)
            CopyVerified(sources.Ex5Path, Path.Combine(ind, IndicatorEx5));
    }

    private static void CopyVerified(string src, string dst)
    {
        var srcLen = new FileInfo(src).Length;
        Exception? last = null;
        for (var attempt = 0; attempt < 4; attempt++)
        {
            try
            {
                File.Copy(src, dst, overwrite: true);
                Thread.Sleep(100);
                var dstInfo = new FileInfo(dst);
                if (!dstInfo.Exists || dstInfo.Length != srcLen)
                    throw new IOException($"Size mismatch after copy: {dst}");
                return;
            }
            catch (Exception ex)
            {
                last = ex;
                if (attempt < 3)
                    Thread.Sleep(1200);
            }
        }

        throw new IOException($"Cannot copy {Path.GetFileName(src)} -> {dst}", last);
    }

    public static IReadOnlyList<string> DiscoverMt5Roots()
    {
        var roots = new List<string>();

        try
        {
            foreach (var p in Process.GetProcessesByName("terminal64"))
            {
                var exe = p.MainModule?.FileName;
                if (string.IsNullOrEmpty(exe)) continue;
                var root = Path.GetDirectoryName(exe);
                if (root is not null && Directory.Exists(Path.Combine(root, "MQL5")))
                    roots.Add(root);
            }
        }
        catch { /* ignore */ }

        if (roots.Count == 0)
        {
            var appData = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "MetaQuotes", "Terminal");
            if (Directory.Exists(appData))
            {
                foreach (var dir in Directory.GetDirectories(appData))
                {
                    var name = Path.GetFileName(dir);
                    if (name.Length == 32 && Directory.Exists(Path.Combine(dir, "MQL5")))
                        roots.Add(dir);
                }
            }

            foreach (var p in new[]
            {
                @"C:\Program Files\WCG Group MT5 Terminal",
                @"C:\Program Files\MetaTrader 5",
                @"D:\Program Files\MetaTrader 5",
            })
            {
                if (Directory.Exists(Path.Combine(p, "MQL5")))
                    roots.Add(p);
            }
        }

        return roots.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
    }

    private static void WriteLog(string message, string level = "INFO")
    {
        var line = $"[{DateTimeOffset.Now:yyyy-MM-dd HH:mm:ss}] [Mt5BridgeDeployer] [{level}] {message}";
        try
        {
            var dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ZhuLong");
            Directory.CreateDirectory(dir);
            File.AppendAllText(Path.Combine(dir, "startup.log"), line + Environment.NewLine);
            File.AppendAllText(Path.Combine(dir, "mt5_deploy.log"), line + Environment.NewLine);
        }
        catch { /* ignore */ }
    }
}
