using System.Runtime.InteropServices;

using Microsoft.Extensions.DependencyInjection;

using Microsoft.Extensions.Logging;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Navigation;
using Serilog;

using ZhuLong.App.Logging;

using ZhuLong.App.Services;

using ZhuLong.App.ViewModels;

using ZhuLong.App.Views;

using ZhuLong.Core;
using ZhuLong.Core.Bootstrap;

using ZhuLong.Core.Configuration;
using ZhuLong.Core.Features;

using ZhuLong.Core.Data;

using ZhuLong.Core.Macro;

using ZhuLong.Core.Pipes;

using ZhuLong.Core.Services;

using ZhuLong.App.Services.Membership;

using Microsoft.EntityFrameworkCore;
using WinRT.Interop;

namespace ZhuLong.App;

public partial class App : Application
{
    /// <summary>供 Inno Setup AppMutex 检测，升级安装前可自动结束进程。</summary>
    public const string InstallerMutexName = "ZhuLong.Trading.App.v3";

    public static IServiceProvider Services { get; private set; } = null!;
    public static Window? MainWindow { get; private set; }

    private Window? _window;
    private static ILoggerFactory? _loggerFactory;
    private static bool _closeToTrayHooked;
    private static int _closeToTrayHookAttempts;
    private static Mutex? _installerMutex;
    private static bool _anotherInstanceRunning;



    public App()

    {
        RuntimeBootstrap.Configure(AppPaths.InstallDir);

        ProcessCrashDiagnostics.InstallOnce();

        UnhandledException += (_, e) =>
        {
            if (IsBenignWinAppRuntimeProbeException(e.Exception))
            {
                e.Handled = true;
                return;
            }

            ProcessCrashDiagnostics.RecordUiException(e.Exception);
            StartupLog.Write("UnhandledException: " + e.Exception);
            e.Handled = true;
        };

        InitializeComponent();

        StartupLog.Write(RuntimeEnvironmentChecker.BuildReport());

        _anotherInstanceRunning = !TryAcquireSingleInstance();

        DevAssetBootstrap.Ensure();

        _loggerFactory = SerilogBootstrap.CreateLoggerFactory();

        Services = ConfigureServices(_loggerFactory);

        _ = AppBootstrap.EnsureFirstRun();

    }



    [DllImport("shell32.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]

    private static extern int SetCurrentProcessExplicitAppUserModelID(string appId);



    private static void TrySetProcessAppUserModelId()

    {

        try { _ = SetCurrentProcessExplicitAppUserModelID("ZhuLong.Trading.ZhuLong.3.1"); }

        catch { /* ignore */ }

    }

    private static bool TryAcquireSingleInstance()
    {
        try
        {
            _installerMutex = new Mutex(true, InstallerMutexName, out var createdNew);
            return createdNew;
        }
        catch (AbandonedMutexException)
        {
            _installerMutex = new Mutex(true, InstallerMutexName, out var createdNew);
            return createdNew;
        }
        catch
        {
            return true;
        }
    }

    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(nint hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern nint FindWindow(string? lpClassName, string? lpWindowName);

    private static void TryActivateExistingInstance()
    {
        try
        {
            var hwnd = FindWindow(null, AppMetadata.FormatVersionLine());
            if (hwnd != 0)
                SetForegroundWindow(hwnd);
        }
        catch { /* ignore */ }
    }



    private static IServiceProvider ConfigureServices(ILoggerFactory loggerFactory)

    {

        var sc = new ServiceCollection();

        sc.AddSingleton(loggerFactory);

        sc.AddLogging(b =>

        {

            b.ClearProviders();

            b.AddSerilog(dispose: false);

            b.AddDebug();

        });

        sc.AddDbContextFactory<ZhuLongDbContext>(o =>

            o.UseSqlite($"Data Source={Core.AppPaths.DatabasePath}"));

        sc.AddSingleton<IApiSecrets, UserSecretsStore>();

        sc.AddSingleton<UserSecretsStore>(sp => (UserSecretsStore)sp.GetRequiredService<IApiSecrets>());

        sc.AddSingleton<IMembershipService>(_ => MembershipHost.Instance);

        sc.AddSingleton<MacroCalendarFetcher>();

        sc.AddSingleton<MacroOfflineRefreshService>();

        sc.AddSingleton<PythonEnvironmentCoordinator>();

        sc.AddSingleton<MacroCalendarService>();

        sc.AddSingleton<PendingSignalStore>();

        sc.AddSingleton<RiskGuardService>();

        sc.AddSingleton<InferenceSnapshotStore>();

        sc.AddSingleton<AlertService>();

        sc.AddSingleton<AttributionService>();

        sc.AddSingleton<PipeServer>(sp =>

        {

            var log = sp.GetRequiredService<ILogger<PipeServer>>();

            var settings = AppSettings.LoadOrCreate(Core.AppPaths.ConfigPath);

            return new PipeServer(log,

                settings.Pipes?.DataPipe ?? @"\\.\pipe\ZhuLong_Data",

                settings.Pipes?.DrawingPipe ?? @"\\.\pipe\ZhuLong_Drawing");

        });

        sc.AddSingleton<PythonGilExecutor>();

        sc.AddSingleton<PythonAgentWorker>();
        sc.AddSingleton<PythonInferenceService>();

        sc.AddSingleton<Mt5ApiWrapper>();

        sc.AddSingleton<DatabaseService>();

        sc.AddSingleton<FeatureCacheService>();

        sc.AddSingleton<MarketSnapshotStore>();

        sc.AddSingleton<PositionManagerService>();

        sc.AddSingleton<ZhuLongRuntimeService>();

        sc.AddSingleton<MainViewModel>();

        return sc.BuildServiceProvider();

    }



    private static bool IsBenignWinAppRuntimeProbeException(Exception? ex) =>
        ex is COMException { HResult: unchecked((int)0x8007007A) };

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        StartupLog.Write($"OnLaunched pid={Environment.ProcessId} anotherInstance={_anotherInstanceRunning}");
        if (_anotherInstanceRunning)
        {
            StartupLog.Write("已有烛龙实例在运行，本进程退出（请查看系统托盘图标，勿重复双击启动）");
            TryActivateExistingInstance();
            Exit();
            return;
        }

        TrySetProcessAppUserModelId();

        _window = new Window { Title = AppMetadata.FormatVersionLine() };
        MainWindow = _window;

        DesktopTrayIcon.ShutdownRuntimeAsync = async () =>
        {
            var runtime = Services.GetRequiredService<ZhuLongRuntimeService>();
            await runtime.StopAsync().ConfigureAwait(false);
            var python = Services.GetService<PythonInferenceService>();
            python?.Dispose();
        };

        if (_window.Content is not Frame rootFrame)
        {
            rootFrame = new Frame();
            rootFrame.NavigationFailed += OnNavigationFailed;
            _window.Content = rootFrame;
        }

        try
        {
            try
            {
                Services.GetRequiredService<DatabaseService>()
                    .EnsureCreatedAsync()
                    .GetAwaiter()
                    .GetResult();
            }
            catch (Exception ex)
            {
                StartupLog.Write("DB schema: " + ex);
            }

            try
            {
                Services.GetRequiredService<ZhuLongRuntimeService>().EnsurePipeServerStarted();
            }
            catch (Exception ex)
            {
                StartupLog.Write("Early pipe listen: " + ex);
            }

            rootFrame.Navigate(typeof(MainPage), "sim-connect");
            StartupLog.Write("OnLaunched: MainPage navigated");

            _window.Activate();
            StartupLog.Write("OnLaunched: window activated");
            ScheduleHookCloseToTray(_window);
            try
            {
                _window.Closed += (_, _) =>
                {
                    if (DesktopTrayIcon.ExitRequested)
                        StartupLog.Write("窗口已关闭：用户从托盘请求退出");
                    else if (_closeToTrayHooked)
                        StartupLog.Write("窗口已隐藏到托盘，推理/管道应仍在运行");
                    else
                        StartupLog.Write("窗口关闭导致进程退出（托盘驻留未启用）");
                };
            }
            catch { /* ignore */ }
        }
        catch (Exception ex)
        {
            StartupLog.Write("OnLaunched: " + ex);
            throw;
        }
    }

    private static void ScheduleHookCloseToTray(Window window)
    {
        void TryAttach(object? sender, WindowActivatedEventArgs? act)
        {
            if (act is not null && act.WindowActivationState == WindowActivationState.Deactivated)
                return;
            if (_closeToTrayHooked)
                return;

            TryHookCloseToTrayCore(window);
            if (!_closeToTrayHooked && _closeToTrayHookAttempts < 40)
            {
                _closeToTrayHookAttempts++;
                var dqRetry = window.DispatcherQueue;
                if (dqRetry is not null)
                    _ = dqRetry.TryEnqueue(DispatcherQueuePriority.Low, () => TryAttach(null, null));
            }
            else if (!_closeToTrayHooked && _closeToTrayHookAttempts >= 40)
            {
                StartupLog.Write("警告：未能挂接「关窗→托盘」；点 × 将直接退出进程（非后台驻留）。请重启后再试或从快捷方式启动。");
            }
        }

        var dq = window.DispatcherQueue;
        if (dq is not null)
            _ = dq.TryEnqueue(DispatcherQueuePriority.Low, () => TryAttach(null, null));
        else
            TryAttach(null, null);

        window.Activated += (s, a) => TryAttach(s, a);
    }

    private static void TryHookCloseToTrayCore(Window window)
    {
        if (_closeToTrayHooked)
            return;

        try
        {
            var hwnd = WindowNative.GetWindowHandle(window);
            if (hwnd == IntPtr.Zero)
                return;

            var windowId = Microsoft.UI.Win32Interop.GetWindowIdFromWindow(hwnd);
            var appWindow = AppWindow.GetFromWindowId(windowId);
            AppBrandAssets.ApplyWindowBranding(appWindow, hwnd);
            appWindow.Closing += (_, args) =>
            {
                if (DesktopTrayIcon.ExitRequested)
                    return;

                args.Cancel = true;
                appWindow.Hide();
                try { DesktopTrayIcon.Ensure(window, appWindow, hwnd); }
                catch { /* ignore */ }
            };
            _closeToTrayHooked = true;
            StartupLog.Write("已启用：关闭窗口时隐藏到托盘（管道/推理继续运行）。");
        }
        catch
        {
            /* 下一帧或 Activated 再试 */
        }
    }



    private void OnNavigationFailed(object sender, NavigationFailedEventArgs e)

    {
        var msg = "Failed to load Page " + e.SourcePageType.FullName + ": " + e.Exception;
        StartupLog.Write("NavigationFailed: " + msg);
        ProcessCrashDiagnostics.RecordUiException(e.Exception, "NavigationFailed");
    }

}



internal static class StartupLog

{

    private static readonly string Path = System.IO.Path.Combine(

        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),

        "ZhuLong", "startup.log");



    public static void Write(string message)

    {

        try

        {

            var dir = System.IO.Path.GetDirectoryName(Path)!;

            Directory.CreateDirectory(dir);

            var line = $"[{DateTimeOffset.Now:yyyy-MM-dd HH:mm:ss}] {message}{Environment.NewLine}";

            File.AppendAllText(Path, line);

        }

        catch { /* ignore */ }

    }

}



/// <summary>开发模式：复制 config/data 到输出目录。</summary>

internal static class DevAssetBootstrap

{

    public static void Ensure()

    {

        var install = Core.AppPaths.InstallDir;

        var root = FindRepoRoot(install);

        if (root is null) return;



        CopyIfMissing(Path.Combine(root, "config.json"), Path.Combine(install, "config.json"));

        CopyDirIfMissing(Path.Combine(root, "data"), Path.Combine(install, "data"));

        CopyDirIfMissing(Path.Combine(root, "models"), Path.Combine(install, "models"));

        CopyDirIfMissing(Path.Combine(root, "ZhuLong.PythonEngine"), Path.Combine(install, "ZhuLong.PythonEngine"));
        CopyDirIfMissing(Path.Combine(root, "mql5"), Path.Combine(install, "mql5"));
        CopyDirIfMissing(Path.Combine(root, "indicators"), Path.Combine(install, "indicators"));

    }



    private static string? FindRepoRoot(string start)

    {

        var dir = start;

        for (var i = 0; i < 8; i++)

        {

            if (File.Exists(Path.Combine(dir, "config.json")) && Directory.Exists(Path.Combine(dir, "zhulong")))

                return dir;

            var parent = Directory.GetParent(dir);

            if (parent is null) return null;

            dir = parent.FullName;

        }

        return null;

    }



    private static void CopyIfMissing(string src, string dst)

    {

        if (!File.Exists(src) || File.Exists(dst)) return;

        Directory.CreateDirectory(Path.GetDirectoryName(dst)!);

        File.Copy(src, dst);

    }



    private static void CopyDirIfMissing(string src, string dst)

    {

        if (!Directory.Exists(src) || Directory.Exists(dst)) return;

        CopyAll(src, dst);

    }



    private static void CopyAll(string src, string dst)

    {

        Directory.CreateDirectory(dst);

        foreach (var file in Directory.GetFiles(src))

            File.Copy(file, Path.Combine(dst, Path.GetFileName(file)), true);

        foreach (var dir in Directory.GetDirectories(src))

            CopyAll(dir, Path.Combine(dst, Path.GetFileName(dir)));

    }

}


