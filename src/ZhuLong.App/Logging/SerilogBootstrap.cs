using Microsoft.Extensions.Logging;
using Serilog;
using ZhuLong.Core;

namespace ZhuLong.App.Logging;

public static class SerilogBootstrap
{
    public static ILoggerFactory CreateLoggerFactory()
    {
        _ = AppPaths.LogsDir;
        var logPath = Path.Combine(AppPaths.LogsDir, "log-.txt");
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .WriteTo.File(
                logPath,
                rollingInterval: RollingInterval.Day,
                retainedFileCountLimit: 14,
                outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff} [{Level:u3}] {SourceContext}{NewLine}{Message:lj}{NewLine}{Exception}")
            .CreateLogger();

        return LoggerFactory.Create(b =>
        {
            b.AddSerilog(dispose: true);
            b.AddDebug();
        });
    }
}
