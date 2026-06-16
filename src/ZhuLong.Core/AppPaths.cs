namespace ZhuLong.Core;

/// <summary>安装目录与 AppData 路径（DELIVERY.md）。</summary>
public static class AppPaths
{
    public static string InstallDir =>
        Path.GetDirectoryName(Environment.ProcessPath) ?? AppContext.BaseDirectory;

    public static string AppDataDir
    {
        get
        {
            var dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "ZhuLong");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    public static string LogsDir
    {
        get
        {
            var dir = Path.Combine(AppDataDir, "logs");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    public static string DatabasePath => Path.Combine(AppDataDir, "trading.db");

    public static string ModelsDir => Path.Combine(InstallDir, "models");

    public static string ModelDir(string symbol) => Path.Combine(ModelsDir, symbol);

    public static string DataDir
    {
        get
        {
            var dir = Path.Combine(InstallDir, "data");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    /// <summary>可写数据目录（AppData）；宏观 JSON 等运行时写入此处。</summary>
    public static string WritableDataDir
    {
        get
        {
            var dir = Path.Combine(AppDataDir, "data");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    public static string MacroEventsPath => Path.Combine(WritableDataDir, "macro_events.csv");

    public static string FredLatestPath => Path.Combine(WritableDataDir, "fred_latest.json");

    public static string SentimentPath => Path.Combine(WritableDataDir, "sentiment.json");

    public static string SecretsDir
    {
        get
        {
            var dir = Path.Combine(AppDataDir, "secrets");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    public static string PythonEngineDir => Path.Combine(InstallDir, "ZhuLong.PythonEngine");

    /// <summary>遗留目录名；烛龙 V3 不捆绑 Python，仅开发机可能仍存在。</summary>
    public static string PythonRuntimeDir => Path.Combine(InstallDir, "python_runtime");

    public static string IndicatorsDir => Path.Combine(InstallDir, "indicators");

    public static string Mql5Dir => Path.Combine(InstallDir, "mql5");

    public static string Mt5PipeDllPath => Path.Combine(Mql5Dir, "Libraries", "ZhuLongMt5Pipe.dll");

    public static string Mt5IndicatorMq5Path => Path.Combine(Mql5Dir, "ZhuLongIndicator.mq5");

    public static string ConfigPath
    {
        get
        {
            var user = Path.Combine(AppDataDir, "config.json");
            if (File.Exists(user)) return user;
            var install = Path.Combine(InstallDir, "config.json");
            return File.Exists(install) ? install : user;
        }
    }

    /// <summary>Serilog 滚动日志：logs/log-yyyyMMdd.txt</summary>
    public static string DailyLogFilePath =>
        Path.Combine(LogsDir, $"log-{DateTime.Now:yyyyMMdd}.txt");

    /// <summary>向上查找含 config.json 与 Python 包的开发/安装根目录。</summary>
    public static string? FindDevRoot()
    {
        var dir = InstallDir;
        for (var i = 0; i < 8; i++)
        {
            if (File.Exists(Path.Combine(dir, "config.json")) &&
                (Directory.Exists(Path.Combine(dir, "zhulong")) ||
                 Directory.Exists(Path.Combine(dir, "ZhuLong.PythonEngine"))))
                return dir;

            var parent = Directory.GetParent(dir);
            if (parent is null) return null;
            dir = parent.FullName;
        }

        return null;
    }

    /// <summary>本机 Python DLL（PYTHONNET_PYDLL / py -3 自动发现）。</summary>
    public static string FindPythonDll()
    {
        var env = Environment.GetEnvironmentVariable("PYTHONNET_PYDLL");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;

        var cached = PythonRuntime.ReadAppDataCache("python_dll.txt");
        if (cached is not null && File.Exists(cached)) return cached;

        var discovered = PythonRuntime.DiscoverDllOnly();
        return discovered ?? "";
    }

    /// <summary>install_python_deps.ps1 绝对路径。</summary>
    public static string PythonDepsScriptPath
    {
        get
        {
            var installScript = Path.Combine(InstallDir, "install_python_deps.ps1");
            if (File.Exists(installScript)) return installScript;
            var devScript = Path.Combine(FindDevRoot() ?? InstallDir, "scripts", "install_python_deps.ps1");
            return File.Exists(devScript) ? devScript : installScript;
        }
    }

    /// <summary>PowerShell 执行示例（含引号，适配 Program Files 路径）。</summary>
    public static string PythonDepsScriptHint => $"& \"{PythonDepsScriptPath}\"";
}
