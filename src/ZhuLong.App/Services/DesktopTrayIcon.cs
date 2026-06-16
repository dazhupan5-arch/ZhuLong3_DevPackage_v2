using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Input;
using H.NotifyIcon;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace ZhuLong.App.Services;

/// <summary>
/// 托盘驻留：关窗隐藏到通知区，双击恢复；右键「退出」才真正结束进程。
/// </summary>
internal static class DesktopTrayIcon
{
    private const int SwRestore = 9;
    private const uint WmClose = 0x0010;

    private static TaskbarIcon? _tray;
    private static Icon? _ownedTrayIcon;
    private static Window? _window;
    private static AppWindow? _appWindow;
    private static nint _hwnd;
    private static XamlRoot? _menuRoot;

    internal static bool ExitRequested { get; private set; }

    /// <summary>托盘「退出」前调用，停止管道/MT5/推理循环。</summary>
    internal static Func<Task>? ShutdownRuntimeAsync { get; set; }

    internal static void TryInstallFromMainPage(Window window, AppWindow appWindow, nint hwnd, XamlRoot menuRoot) =>
        Ensure(window, appWindow, hwnd, menuRoot);

    internal static void Ensure(Window window, AppWindow appWindow, nint hwnd, XamlRoot? menuRoot = null)
    {
        _window = window;
        _appWindow = appWindow;
        _hwnd = hwnd;
        if (menuRoot is not null)
            _menuRoot = menuRoot;

        if (_tray is not null && _tray.IsCreated)
        {
            if (_menuRoot is not null)
            {
                try { EnsureContextMenu(_menuRoot); }
                catch { /* ignore */ }
            }
            return;
        }

        if (_tray is not null)
        {
            try { _tray.Dispose(); }
            catch { /* ignore */ }
            _tray = null;
        }

        ReleaseOwnedTrayIcon();

        if (menuRoot is null && window.Content is FrameworkElement fe && fe.XamlRoot is null)
        {
            void OnLoaded(object sender, RoutedEventArgs args)
            {
                fe.Loaded -= OnLoaded;
                Ensure(window, appWindow, hwnd, menuRoot: null);
            }

            fe.Loaded += OnLoaded;
            return;
        }

        _tray = new TaskbarIcon
        {
            ToolTipText = "烛龙 ZhuLong · 双击打开窗口 · 右键菜单",
            DoubleClickCommand = new SimpleCommand(ShowMainWindow),
        };

        _ownedTrayIcon = LoadTrayIconBitmap();
        if (_ownedTrayIcon is not null)
            _tray.UpdateIcon(_ownedTrayIcon);

        var xr = _menuRoot ?? menuRoot ?? (window.Content as FrameworkElement)?.XamlRoot;
        EnsureContextMenu(xr);

        try
        {
            _tray.ForceCreate(false);
        }
        catch (Exception ex)
        {
            StartupLog.Write("托盘图标 ForceCreate 失败: " + ex.Message);
            _tray.ContextFlyout = null;
            try { _tray.ForceCreate(false); }
            catch (Exception ex2)
            {
                StartupLog.Write("托盘图标 ForceCreate(无菜单) 失败: " + ex2.Message);
            }
        }

        if (_ownedTrayIcon is not null)
        {
            try { _tray.UpdateIcon(_ownedTrayIcon); }
            catch { /* ignore */ }
        }

        if (!_tray.IsCreated)
        {
            try { _tray.Dispose(); }
            catch { /* ignore */ }
            _tray = null;
            ReleaseOwnedTrayIcon();
        }
    }

    private static void EnsureContextMenu(XamlRoot? xr)
    {
        if (_tray is null) return;

        if (xr is null)
        {
            _tray.ContextFlyout = null;
            return;
        }

        var flyout = new MenuFlyout { XamlRoot = xr };
        var openItem = new MenuFlyoutItem { Text = "打开主窗口", Command = new SimpleCommand(ShowMainWindow) };
        openItem.Click += (_, _) => ShowMainWindow();
        var exitItem = new MenuFlyoutItem { Text = "退出烛龙", Command = new SimpleCommand(QuitFromTray) };
        exitItem.Click += (_, _) => QuitFromTray();
        flyout.Items.Add(openItem);
        flyout.Items.Add(exitItem);
        _tray.ContextFlyout = flyout;
    }

    private static void ReleaseOwnedTrayIcon()
    {
        if (_ownedTrayIcon is null) return;
        try { _ownedTrayIcon.Dispose(); }
        catch { /* ignore */ }
        _ownedTrayIcon = null;
    }

    private static Icon? LoadTrayIconBitmap()
    {
        var branded = AppBrandAssets.LoadTrayIcon();
        if (branded is not null)
            return branded;

        foreach (var rel in new[] { AppBrandAssets.WindowIconRelative, AppBrandAssets.AppIconRelative })
        {
            var path = AppBrandAssets.ResolvePath(rel);
            if (string.IsNullOrEmpty(path))
                continue;

            try
            {
                using var tmp = new Icon(path, 32, 32);
                return (Icon)tmp.Clone();
            }
            catch
            {
                try
                {
                    using var tmp = new Icon(path, 16, 16);
                    return (Icon)tmp.Clone();
                }
                catch { /* next */ }
            }
        }

        try
        {
            var ep = Environment.ProcessPath;
            if (!string.IsNullOrEmpty(ep))
            {
                using var ext = Icon.ExtractAssociatedIcon(ep);
                if (ext is not null)
                    return (Icon)ext.Clone();
            }
        }
        catch { /* ignore */ }

        return null;
    }

    private static IEnumerable<string> EnumerateAppBaseDirectories()
    {
        var h = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        void Add(string? s)
        {
            if (string.IsNullOrWhiteSpace(s)) return;
            try
            {
                var full = Path.GetFullPath(s);
                if (Directory.Exists(full))
                    h.Add(full);
            }
            catch { /* skip */ }
        }

        Add(AppContext.BaseDirectory);
        Add(AppDomain.CurrentDomain.BaseDirectory);
        try { Add(Path.GetDirectoryName(Environment.ProcessPath)); }
        catch { /* ignore */ }

        return h;
    }

    private static void ShowMainWindow()
    {
        var w = _window;
        var aw = _appWindow;
        if (w is null || aw is null) return;

        var dq = w.DispatcherQueue;
        if (dq is null) return;

        void Core()
        {
            try
            {
                aw.Show();
                _ = ShowWindow(_hwnd, SwRestore);
                _ = SetForegroundWindow(_hwnd);
            }
            catch { /* ignore */ }
        }

        if (dq.HasThreadAccess)
            Core();
        else
            _ = dq.TryEnqueue(DispatcherQueuePriority.Normal, Core);
    }

    private static void QuitFromTray()
    {
        ExitRequested = true;
        StartupLog.Write("托盘退出：请求停止运行时并退出进程");
        StartForceExitFallback();

        var dq = _window?.DispatcherQueue ?? App.MainWindow?.DispatcherQueue;

        async void CoreAsync()
        {
            try
            {
                if (ShutdownRuntimeAsync is not null)
                    await ShutdownRuntimeAsync().ConfigureAwait(true);
            }
            catch (Exception ex)
            {
                StartupLog.Write("托盘退出：停止运行时异常 " + ex.Message);
            }

            try
            {
                if (_tray is not null)
                {
                    _tray.Dispose();
                    _tray = null;
                }
            }
            catch { /* ignore */ }

            ReleaseOwnedTrayIcon();

            try
            {
                if (_window is not null)
                {
                    MarkGracefulExitStarted();
                    _window.Close();
                    return;
                }
            }
            catch { /* fall through */ }

            try
            {
                MarkGracefulExitStarted();
                Application.Current.Exit();
            }
            catch
            {
                MarkGracefulExitStarted();
                Environment.Exit(0);
            }
        }

        if (dq is not null)
        {
            if (dq.HasThreadAccess)
                CoreAsync();
            else if (!dq.TryEnqueue(DispatcherQueuePriority.High, CoreAsync))
                RequestCloseViaWin32();
            return;
        }

        RequestCloseViaWin32();

        void RequestCloseViaWin32()
        {
            if (_hwnd != 0 && PostMessage(_hwnd, WmClose, UIntPtr.Zero, IntPtr.Zero))
                return;

            try { Application.Current.Exit(); }
            catch { /* ignore */ }
            Environment.Exit(0);
        }
    }

    private static int _forceExitOnce;
    private static int _gracefulExitSignaled;

    private static void StartForceExitFallback()
    {
        if (Interlocked.Exchange(ref _forceExitOnce, 1) != 0)
            return;

        _ = Task.Run(async () =>
        {
            try { await Task.Delay(15000).ConfigureAwait(false); }
            catch { /* ignore */ }

            if (Interlocked.CompareExchange(ref _gracefulExitSignaled, 0, 0) != 0)
                return;

            StartupLog.Write("托盘退出：15s 内未正常结束，强制结束进程");
            try { Process.GetCurrentProcess().Kill(entireProcessTree: true); }
            catch { Environment.Exit(0); }
        });
    }

    private static void MarkGracefulExitStarted() =>
        Interlocked.Exchange(ref _gracefulExitSignaled, 1);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool PostMessage(nint hWnd, uint msg, UIntPtr wParam, IntPtr lParam);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool ShowWindow(nint hWnd, int nCmdShow);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool SetForegroundWindow(nint hWnd);

    private sealed class SimpleCommand : ICommand
    {
        private readonly Action _action;
        public SimpleCommand(Action action) => _action = action;
#pragma warning disable CS0067
        public event EventHandler? CanExecuteChanged;
#pragma warning restore CS0067
        public bool CanExecute(object? parameter) => true;
        public void Execute(object? parameter) => _action();
    }
}
