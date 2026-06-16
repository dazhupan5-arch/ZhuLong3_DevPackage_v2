using Microsoft.Extensions.DependencyInjection;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Navigation;
using WinRT.Interop;
using ZhuLong.App.Services;
using ZhuLong.App.Services.Membership;
using ZhuLong.App.ViewModels;
using ZhuLong.Core.Models;

namespace ZhuLong.App.Views;

public sealed partial class MainPage : Page
{
    public MainViewModel ViewModel { get; }

    public MainPage()
    {
        ViewModel = App.Services.GetRequiredService<MainViewModel>();
        InitializeComponent();
        DataContext = ViewModel;
        ViewModel.BindUiDispatcher(DispatcherQueue);
        MembershipHost.Instance.Refresh();
        ApplyBrandAssets();
        Loaded += OnMainPageLoaded;
        ViewModel.PropertyChanged += async (_, e) =>
        {
            if (e.PropertyName == nameof(MainViewModel.ModelDialogMessage) && ViewModel.ModelDialogMessage is { } msg)
            {
                ViewModel.ModelDialogMessage = null;
                var dlg = new ContentDialog
                {
                    Title = "模型文件缺失",
                    Content = msg,
                    CloseButtonText = "知道了",
                    XamlRoot = XamlRoot,
                };
                await dlg.ShowAsync();
            }
        };
    }

    private void ApplyBrandAssets()
    {
        AppBrandAssets.ApplyTitleImages(TitleLogoImage, PaneLogoImage);
    }

    private void OnMainPageLoaded(object sender, RoutedEventArgs e)
    {
        var dq = DispatcherQueue;
        if (dq is null || App.MainWindow is null || XamlRoot is null)
            return;

        _ = dq.TryEnqueue(DispatcherQueuePriority.Low, () =>
        {
            _ = dq.TryEnqueue(DispatcherQueuePriority.Low, () =>
            {
                try
                {
                    if (App.MainWindow is null || XamlRoot is null)
                        return;

                    var hwnd = WindowNative.GetWindowHandle(App.MainWindow);
                    if (hwnd == IntPtr.Zero)
                        return;

                    var windowId = Microsoft.UI.Win32Interop.GetWindowIdFromWindow(hwnd);
                    var appWindow = AppWindow.GetFromWindowId(windowId);
                    AppBrandAssets.ApplyWindowBranding(appWindow, hwnd);
                    DesktopTrayIcon.TryInstallFromMainPage(App.MainWindow, appWindow, hwnd, XamlRoot);
                }
                catch
                {
                    /* 托盘失败不阻塞主界面 */
                }
            });
        });
    }

    protected override async void OnNavigatedTo(NavigationEventArgs e)
    {
        var autoConnect = e.Parameter is true
            || string.Equals(e.Parameter as string, "sim-connect", StringComparison.OrdinalIgnoreCase);
        if (autoConnect)
            await ViewModel.ConnectAsync();
        await ViewModel.RefreshSignalsCommand.ExecuteAsync(null);
    }

    private void OnNavSelectionChanged(NavigationView sender, NavigationViewSelectionChangedEventArgs args)
    {
        if (args.SelectedItem is not NavigationViewItem item || item.Tag is not string tag)
            return;

        ViewModel.SelectedPanel = tag;
        SignalsPanel.Visibility = tag == "signals" ? Visibility.Visible : Visibility.Collapsed;
        PositionsPanel.Visibility = tag == "positions" ? Visibility.Visible : Visibility.Collapsed;
        ParamsPanel.Visibility = tag == "params" ? Visibility.Visible : Visibility.Collapsed;
        SettingsPanel.Visibility = tag == "settings" ? Visibility.Visible : Visibility.Collapsed;
        AnalysisPanel.Visibility = tag == "analysis" ? Visibility.Visible : Visibility.Collapsed;
        if (tag == "positions")
            ViewModel.RefreshPositionsCommand.Execute(null);
        if (tag == "params")
            ParamsPanel.LoadFromDisk();
        if (tag == "settings")
            SettingsPanel.RefreshUi();
        if (tag == "analysis")
            _ = AnalysisPanel.RefreshAsync();
    }

    private void OnSignalClick(object sender, ItemClickEventArgs e)
    {
        if (e.ClickedItem is SignalModel signal)
            ViewModel.CopyCommentCommand.Execute(signal);
    }
}
