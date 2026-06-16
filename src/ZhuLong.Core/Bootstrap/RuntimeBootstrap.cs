namespace ZhuLong.Core.Bootstrap;

/// <summary>进程启动最早阶段：绑定本机 .NET 根目录，避免 D 盘安装时 apphost 误判缺运行库。</summary>
public static class RuntimeBootstrap
{
    public static DotNetRuntimeDiscovery.DesktopRuntimeInfo Configure(string installDir)
    {
        EnsureRuntimeConfig(installDir);

        var info = DotNetRuntimeDiscovery.ProbeDesktopRuntime();
        if (info.IsReady)
            DotNetRuntimeDiscovery.ApplyEnvironment(info.InstallRoot);
        else
            DotNetRuntimeDiscovery.ApplyEnvironment(DotNetRuntimeDiscovery.ReadCachedInstallRoot());

        return info;
    }

    private static void EnsureRuntimeConfig(string installDir)
    {
        try
        {
            var path = Path.Combine(installDir, "ZhuLong.runtimeconfig.json");
            if (!File.Exists(path)) return;

            var text = File.ReadAllText(path);
            if (text.Contains("Microsoft.WindowsDesktop.App", StringComparison.Ordinal)
                && text.Contains("LatestPatch", StringComparison.Ordinal))
                return;

            var fix = Path.Combine(installDir, "scripts", "fix_runtimeconfig.ps1");
            if (!File.Exists(fix)) return;

            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "powershell.exe",
                Arguments = $"-NoProfile -ExecutionPolicy Bypass -File \"{fix}\" -StageDir \"{installDir}\"",
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using var p = System.Diagnostics.Process.Start(psi);
            p?.WaitForExit(15000);
        }
        catch
        {
            /* 不阻断启动 */
        }
    }
}
