using System.Diagnostics;

namespace ZhuLong.Core;

/// <summary>本机 Python 3 路径解析（不捆绑 python_runtime）。</summary>
public static class PythonRuntime
{
    private static string? _cachedExe;
    private static string? _cachedDll;

    public static void InvalidateCache()
    {
        _cachedExe = null;
        _cachedDll = null;
    }

    public static string ResolveExecutable()
    {
        if (_cachedExe is not null && File.Exists(_cachedExe))
            return _cachedExe;

        var cached = ReadAppDataCache("python_exe.txt");
        if (cached is not null && File.Exists(cached))
            return _cachedExe = cached;

        var discovered = DiscoverViaLauncher();
        return _cachedExe = discovered.exe ?? "python";
    }

    public static string ResolvePythonDll() => AppPaths.FindPythonDll();

    public static (string? exe, string? dll) DiscoverAndCache(bool force = false)
    {
        if (force)
            InvalidateCache();

        var result = DiscoverViaLauncher();
        if (result.exe is not null)
            WriteAppDataCache("python_exe.txt", result.exe);
        if (result.dll is not null)
            WriteAppDataCache("python_dll.txt", result.dll);
        return result;
    }

    private static (string? exe, string? dll) DiscoverViaLauncher()
    {
        foreach (var starter in new[] { "python", "py", "python3" })
        {
            if (!PythonExecutableResolver.TryResolve(starter, out var exe, out _))
                continue;

            try
            {
                var code = """
                    import sys, os
                    base = getattr(sys, 'base_prefix', sys.prefix)
                    dll = os.path.join(base, f'python{sys.version_info.major}{sys.version_info.minor}.dll')
                    print(sys.executable)
                    print(dll)
                    """;

                var psi = new ProcessStartInfo
                {
                    FileName = exe,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                };
                if (PythonQuickProbe.IsPyLauncher(exe))
                    psi.ArgumentList.Add("-3");
                psi.ArgumentList.Add("-c");
                psi.ArgumentList.Add(code);

                using var p = Process.Start(psi);
                if (p is null) continue;
                var output = p.StandardOutput.ReadToEnd();
                if (!p.WaitForExit(15000) || p.ExitCode != 0) continue;

                var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                if (lines.Length < 2) continue;

                var resolvedExe = lines[0];
                var dll = lines[1];
                if (!File.Exists(resolvedExe) || !File.Exists(dll)) continue;

                WriteAppDataCache("python_exe.txt", resolvedExe);
                WriteAppDataCache("python_dll.txt", dll);
                Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", dll);
                _cachedExe = resolvedExe;
                _cachedDll = dll;
                return (resolvedExe, dll);
            }
            catch
            {
                /* try next */
            }
        }

        return (null, null);
    }

    internal static string? ReadAppDataCache(string fileName)
    {
        var path = Path.Combine(AppPaths.AppDataDir, fileName);
        if (!File.Exists(path)) return null;
        var text = File.ReadAllText(path).Trim();
        return string.IsNullOrEmpty(text) ? null : text;
    }

    internal static void WriteAppDataCache(string fileName, string value)
    {
        try
        {
            Directory.CreateDirectory(AppPaths.AppDataDir);
            File.WriteAllText(Path.Combine(AppPaths.AppDataDir, fileName), value);
        }
        catch
        {
            /* ignore */
        }
    }

    internal static string? DiscoverDllOnly()
    {
        if (_cachedDll is not null && File.Exists(_cachedDll))
            return _cachedDll;

        var cached = ReadAppDataCache("python_dll.txt");
        if (cached is not null && File.Exists(cached))
            return _cachedDll = cached;

        var (_, dll) = DiscoverViaLauncher();
        return _cachedDll = dll;
    }
}
