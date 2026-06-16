using System.Diagnostics;
using Microsoft.Win32;

namespace ZhuLong.Core.Bootstrap;

/// <summary>发现本机 .NET 安装根目录（与安装盘无关，扫描注册表 / 环境变量 / 标准路径）。</summary>
public static class DotNetRuntimeDiscovery
{
    public const string DesktopFramework = "Microsoft.WindowsDesktop.App";
    public const string RequiredMajor = "8.0";

    public sealed record DesktopRuntimeInfo(string InstallRoot, IReadOnlyList<string> DesktopVersions)
    {
        public bool IsReady => DesktopVersions.Count > 0;
    }

    public static DesktopRuntimeInfo ProbeDesktopRuntime()
    {
        foreach (var root in EnumerateInstallRoots())
        {
            var versions = FindFrameworkVersions(root, DesktopFramework, RequiredMajor);
            if (versions.Count > 0)
                return new DesktopRuntimeInfo(root, versions);
        }

        return ProbeViaDotnetListRuntimes() ?? new DesktopRuntimeInfo("", []);
    }

    public static void ApplyEnvironment(string? installRoot)
    {
        Environment.SetEnvironmentVariable("DOTNET_ROLL_FORWARD", "LatestPatch");
        Environment.SetEnvironmentVariable("DOTNET_MULTILEVEL_LOOKUP", "1");
        if (string.IsNullOrWhiteSpace(installRoot))
            return;

        var root = installRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        Environment.SetEnvironmentVariable("DOTNET_ROOT", root);
        Environment.SetEnvironmentVariable("DOTNET_ROOT(x86)", root);

        var path = Environment.GetEnvironmentVariable("PATH") ?? "";
        if (!path.Contains(root, StringComparison.OrdinalIgnoreCase))
            Environment.SetEnvironmentVariable("PATH", root + Path.PathSeparator + path);

        try
        {
            Directory.CreateDirectory(AppPaths.AppDataDir);
            File.WriteAllText(Path.Combine(AppPaths.AppDataDir, "dotnet_root.txt"), root);
        }
        catch
        {
            /* ignore */
        }
    }

    public static string? ReadCachedInstallRoot()
    {
        try
        {
            var path = Path.Combine(AppPaths.AppDataDir, "dotnet_root.txt");
            if (!File.Exists(path)) return null;
            var text = File.ReadAllText(path).Trim();
            return string.IsNullOrEmpty(text) ? null : text;
        }
        catch
        {
            return null;
        }
    }

    private static IEnumerable<string> EnumerateInstallRoots()
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        void TryAdd(string? path)
        {
            if (string.IsNullOrWhiteSpace(path)) return;
            var full = Path.GetFullPath(path.TrimEnd('\\', '/'));
            if (!Directory.Exists(full)) return;
            seen.Add(full);
        }

        TryAdd(Environment.GetEnvironmentVariable("DOTNET_ROOT"));
        TryAdd(ReadCachedInstallRoot());
        TryAdd(ReadRegistryInstallLocation(@"SOFTWARE\dotnet\Setup\InstalledVersions\x64\InstallLocation"));
        TryAdd(ReadRegistryInstallLocation(@"SOFTWARE\WOW6432Node\dotnet\Setup\InstalledVersions\x64\InstallLocation"));

        foreach (var pf in new[]
                 {
                     Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                     Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86),
                 })
        {
            if (!string.IsNullOrEmpty(pf))
                TryAdd(Path.Combine(pf, "dotnet"));
        }

        var where = TryWhereDotnetExe();
        if (where is not null)
        {
            var dir = Path.GetDirectoryName(where);
            if (!string.IsNullOrEmpty(dir))
                TryAdd(dir);
        }

        return seen;
    }

    private static string? ReadRegistryInstallLocation(string subKey)
    {
        try
        {
            using var key = Registry.LocalMachine.OpenSubKey(subKey);
            return key?.GetValue("InstallLocation") as string;
        }
        catch
        {
            return null;
        }
    }

    private static string? TryWhereDotnetExe()
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "where.exe",
                Arguments = "dotnet",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                CreateNoWindow = true,
            };
            using var p = Process.Start(psi);
            if (p is null) return null;
            var line = p.StandardOutput.ReadLine()?.Trim();
            p.WaitForExit(3000);
            return p.ExitCode == 0 && !string.IsNullOrEmpty(line) && File.Exists(line) ? line : null;
        }
        catch
        {
            return null;
        }
    }

    private static List<string> FindFrameworkVersions(string installRoot, string framework, string majorPrefix)
    {
        var list = new List<string>();
        var dir = Path.Combine(installRoot, "shared", framework);
        if (!Directory.Exists(dir)) return list;
        foreach (var sub in Directory.GetDirectories(dir))
        {
            var name = Path.GetFileName(sub);
            if (name.StartsWith(majorPrefix + ".", StringComparison.Ordinal))
                list.Add(name);
        }
        return list.Distinct(StringComparer.Ordinal).OrderBy(v => v).ToList();
    }

    private static DesktopRuntimeInfo? ProbeViaDotnetListRuntimes()
    {
        foreach (var root in EnumerateInstallRoots())
        {
            var dotnet = Path.Combine(root, "dotnet.exe");
            if (!File.Exists(dotnet)) continue;
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = dotnet,
                    Arguments = "--list-runtimes",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true,
                };
                using var p = Process.Start(psi);
                if (p is null) continue;
                var output = p.StandardOutput.ReadToEnd();
                if (!p.WaitForExit(8000) || p.ExitCode != 0) continue;

                var versions = output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Where(l => l.Contains(DesktopFramework + " " + RequiredMajor + ".", StringComparison.Ordinal))
                    .Select(l => l.Split(' ', StringSplitOptions.RemoveEmptyEntries).ElementAtOrDefault(1) ?? "")
                    .Where(v => v.StartsWith(RequiredMajor + ".", StringComparison.Ordinal))
                    .Distinct(StringComparer.Ordinal)
                    .OrderBy(v => v)
                    .ToList();
                if (versions.Count > 0)
                    return new DesktopRuntimeInfo(root, versions);
            }
            catch
            {
                /* try next root */
            }
        }

        return null;
    }
}
