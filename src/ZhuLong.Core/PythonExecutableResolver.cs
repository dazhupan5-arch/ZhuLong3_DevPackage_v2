using System.Diagnostics;

namespace ZhuLong.Core;

/// <summary>
/// 解析 python/py 为可执行文件全路径。WinUI 子进程常拿不到与用户 CMD 相同的 PATH，故增加 where 与常见安装目录回退。
/// </summary>
public static class PythonExecutableResolver
{
    public static string Resolve(string configured = "python")
    {
        var c = (configured ?? string.Empty).Trim();
        if (string.IsNullOrEmpty(c))
            c = "python";

        if (c.IndexOf(Path.DirectorySeparatorChar) >= 0 || c.IndexOf(Path.AltDirectorySeparatorChar) >= 0)
        {
            if (!File.Exists(c))
                throw new FileNotFoundException("Python 路径不存在: " + c, c);
            return Path.GetFullPath(c);
        }

        if (!OperatingSystem.IsWindows())
            return c;

        // 与 DiscoverAndCache / pip 安装使用同一解释器，避免 py.exe 与 python.exe 不一致
        if (c.Equals("python", StringComparison.OrdinalIgnoreCase) ||
            c.Equals("py", StringComparison.OrdinalIgnoreCase))
        {
            var cached = PythonRuntime.ReadAppDataCache("python_exe.txt");
            if (!string.IsNullOrEmpty(cached) && File.Exists(cached))
                return Path.GetFullPath(cached);
        }

        var viaWhere = TryWhereOnPath(c);
        if (!string.IsNullOrEmpty(viaWhere))
            return viaWhere;

        if (c.Equals("python", StringComparison.OrdinalIgnoreCase))
        {
            viaWhere = TryWhereOnPath("py");
            if (!string.IsNullOrEmpty(viaWhere))
                return viaWhere;
        }

        var guessed = TryFindUnderPythonInstallDirs();
        if (!string.IsNullOrEmpty(guessed))
            return guessed;

        throw new FileNotFoundException(
            "未找到 Python。请安装 Python 3.10+ 并勾选 Add to PATH，或在设置页点「一键修复环境」。",
            c);
    }

    public static bool TryResolve(string configured, out string? resolved, out string? error)
    {
        try
        {
            resolved = Resolve(configured);
            error = null;
            return true;
        }
        catch (Exception ex)
        {
            resolved = null;
            error = ex.Message;
            return false;
        }
    }

    private static string? TryWhereOnPath(string commandName)
    {
        try
        {
            var sysRoot = Environment.GetEnvironmentVariable("SystemRoot");
            if (string.IsNullOrEmpty(sysRoot))
                return null;

            var whereExe = Path.Combine(sysRoot, "System32", "where.exe");
            if (!File.Exists(whereExe))
                return null;

            var psi = new ProcessStartInfo
            {
                FileName = whereExe,
                Arguments = commandName,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };

            using var p = Process.Start(psi);
            if (p is null)
                return null;

            var stdout = p.StandardOutput.ReadToEnd();
            _ = p.WaitForExit(10000);

            foreach (var raw in stdout.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries))
            {
                var line = raw.Trim();
                if (string.IsNullOrEmpty(line) || !File.Exists(line))
                    continue;
                if (line.Contains("\\WindowsApps\\", StringComparison.OrdinalIgnoreCase) ||
                    line.Contains("/WindowsApps/", StringComparison.OrdinalIgnoreCase))
                    continue;
                return line;
            }
        }
        catch
        {
            /* ignore */
        }

        return null;
    }

    private static string? TryFindUnderPythonInstallDirs()
    {
        try
        {
            var local = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Programs", "Python");
            var found = ScanPythonRoot(local);
            if (!string.IsNullOrEmpty(found))
                return found;

            foreach (var special in new[] { Environment.SpecialFolder.ProgramFiles, Environment.SpecialFolder.ProgramFilesX86 })
            {
                var pf = Environment.GetFolderPath(special);
                if (string.IsNullOrEmpty(pf))
                    continue;

                foreach (var leaf in new[] { "Python313", "Python312", "Python311", "Python310", "Python39" })
                {
                    var p = Path.Combine(pf, leaf, "python.exe");
                    if (File.Exists(p))
                        return p;
                }
            }
        }
        catch
        {
            /* ignore */
        }

        return null;
    }

    private static string? ScanPythonRoot(string root)
    {
        if (!Directory.Exists(root))
            return null;

        foreach (var dir in Directory.GetDirectories(root))
        {
            var direct = Path.Combine(dir, "python.exe");
            if (File.Exists(direct))
                return direct;

            try
            {
                foreach (var sub in Directory.GetDirectories(dir))
                {
                    var nested = Path.Combine(sub, "python.exe");
                    if (File.Exists(nested))
                        return nested;
                }
            }
            catch
            {
                /* ignore */
            }
        }

        return null;
    }
}
