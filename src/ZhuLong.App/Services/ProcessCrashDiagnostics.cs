using System.Runtime.InteropServices;
using System.Text;
using ZhuLong.Core;

namespace ZhuLong.App.Services;

/// <summary>捕获非 UI 线程未处理异常与未观察 Task 异常，写入本地日志（避免进程静默退出后无迹可寻）。</summary>
public static class ProcessCrashDiagnostics
{
    private static int _installed;

    public static void InstallOnce()
    {
        if (Interlocked.Exchange(ref _installed, 1) != 0)
            return;

        AppDomain.CurrentDomain.UnhandledException += OnAppDomainUnhandled;
        AppDomain.CurrentDomain.ProcessExit += (_, _) =>
            StartupLog.Write("ProcessExit：进程即将结束（若未点托盘退出，请查看 process_crash.log）");
        TaskScheduler.UnobservedTaskException += OnUnobservedTaskException;
    }

    public static void RecordUiException(Exception? ex, string source = "UI.UnhandledException")
    {
        if (IsBenignWinAppRuntimeProbeException(ex))
            return;

        Record(FormatException(source, ex, isTerminating: false));
    }

    private static bool IsBenignWinAppRuntimeProbeException(Exception? ex) =>
        ex is COMException { HResult: unchecked((int)0x8007007A) };

    private static void OnAppDomainUnhandled(object sender, System.UnhandledExceptionEventArgs e)
    {
        var ex = e.ExceptionObject as Exception;
        Record(FormatException("AppDomain.UnhandledException", ex, e.IsTerminating));
        if (ex is AccessViolationException or SEHException)
            StartupLog.Write("致命: 原生模块崩溃(AccessViolation)，请升级 v1.0.8+ 并完全退出后重开再连 MT5");
    }

    private static void OnUnobservedTaskException(object? sender, UnobservedTaskExceptionEventArgs e)
    {
        Record(FormatException("TaskScheduler.UnobservedTaskException", e.Exception, isTerminating: false));
        try { e.SetObserved(); }
        catch { /* ignore */ }
    }

    private static string FormatException(string source, Exception? ex, bool isTerminating)
    {
        var sb = new StringBuilder(512);
        sb.Append('[').Append(DateTimeOffset.Now.ToString("yyyy-MM-dd HH:mm:ss")).Append("] ");
        sb.Append(source);
        if (isTerminating)
            sb.Append(" terminating=true");
        sb.AppendLine();

        if (ex is null)
        {
            sb.AppendLine("(no Exception object)");
            return sb.ToString();
        }

        for (var cur = ex; cur is not null; cur = cur.InnerException)
        {
            sb.AppendLine(cur.GetType().FullName + ": " + cur.Message);
            if (cur is COMException com)
                sb.AppendLine("HResult=0x" + com.HResult.ToString("X8"));
            var trace = cur.StackTrace;
            sb.AppendLine(string.IsNullOrWhiteSpace(trace)
                ? new System.Diagnostics.StackTrace(true).ToString()
                : trace);
        }

        return sb.ToString();
    }

    private static void Record(string message)
    {
        var oneLine = message.Length > 2000 ? message[..2000] + "…" : message;
        StartupLog.Write(oneLine.Replace('\r', ' ').Replace('\n', ' '));

        try
        {
            var path = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "ZhuLong", "process_crash.log");
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.AppendAllText(path, message + Environment.NewLine + "----" + Environment.NewLine, Encoding.UTF8);
        }
        catch { /* ignore */ }

        try
        {
            var logsDir = AppPaths.LogsDir;
            File.AppendAllText(
                Path.Combine(logsDir, "crash.log"),
                message + Environment.NewLine + "----" + Environment.NewLine,
                Encoding.UTF8);
        }
        catch { /* ignore */ }
    }
}
